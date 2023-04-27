import json
import random
from datasets import Dataset
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments



j = open("dataset/CONAN/CONAN.json", "r")
conan = json.load(j)
english_conan = list(filter(lambda cn: cn["cn_id"].startswith("EN"), conan["conan"]))



model_name = "google/flan-t5-xl"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

cuda = torch.device("cuda")



j = open("dataset/CONAN/CONAN.json", "r")
conan = json.load(j)
english_conan = list(filter(lambda cn: cn["cn_id"].startswith("EN"), conan["conan"]))

group_by_tweet = {}
for cn in english_conan:
    if not cn["hateSpeech"] in group_by_tweet:
        group_by_tweet[cn["hateSpeech"]] = [cn["counterSpeech"]]
    else:
        group_by_tweet[cn["hateSpeech"]].append(cn["counterSpeech"])

acum = 0
train_threshold = len(english_conan) * 0.7
val_threshold = len(english_conan) * 0.8
train_dataset = []
val_dataset = []
test_dataset = []
keys = list(group_by_tweet.keys())
random.seed(42)
random.shuffle(keys)
for key in keys:
    for cn in group_by_tweet[key]:
        to_append = {"hateSpeech": key, "counterSpeech": cn}
        print(to_append)
        if acum < train_threshold:
            train_dataset.append(to_append)
        elif acum < val_threshold:
            val_dataset.append(to_append)
        else:
            test_dataset.append(to_append)

    acum += len(group_by_tweet[key])
train_data = Dataset.from_pandas(pd.DataFrame(train_dataset))
val_data = Dataset.from_pandas(pd.DataFrame(val_dataset))
test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))

# assert(False)

with torch.cuda.device('cuda:0'):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    print("We added ", num_new_tokens, " new tokens")

    model.resize_token_embeddings(len(tokenizer))

    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Linear):
        def forward(self, x): return super().forward(x).to(torch.device.float32)
    
    model.lm_head=CastOutputToFloat(model.lm_head.in_features, model.lm_head.out_features, bias = model.lm_head.bias)

    def print_trainable_parameters(model):

        trainable_params = 0
        all_params = 0
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100* (trainable_params / all_params)}"
        )
    
    config = LoraConfig(
        r=16,
        lora_alpha = 32,
        target_modules=["q", "v"],
        lora_dropout = 0.05,
        bias="none"
    )


    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    model.config.use_cache = False

    dataset_tokenized = list(map(lambda sample: tokenizer("Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => ", truncation=True)["input_ids"], english_conan))
    max_source_length = max([len(x) for x in dataset_tokenized])

    target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], english_conan))
    max_target_length = max([len(x) for x in target_tokenized])

    def preprocess(sample, padding="max_length"):
        inputs = "Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => "
        # labels = [sample["counterSpeech"]]
        model_inputs = tokenizer(inputs, padding=padding, max_length=max_source_length, truncation=True, return_tensors="pt")
        labels = tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", padding=padding, max_length=max_target_length, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [
                (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
            ]
        model_inputs["labels"] = [labels["input_ids"]]
        return model_inputs

    # random.Random(42).shuffle(english_conan)
    train_data = train_data.map(preprocess, remove_columns=["hateSpeech", "counterSpeech"])
    val_data = val_data.map(preprocess, remove_columns=["hateSpeech", "counterSpeech"])
    test_data = test_data.map(preprocess, remove_columns=["hateSpeech", "counterSpeech"])

    print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
    print(train_data[0])
    # Metric
    metric = evaluate.load("bertscore")

    # helper function to postprocess text
    def postprocess_text(preds, labels, inputs):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        inputs = [input.strip() for input in inputs]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
        inputs = ["\n".join(sent_tokenize(input)) for input in inputs]

        return preds, labels, inputs

    def compute_metrics(eval_preds):
        preds, labels, inputs = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels, decoded_inputs = postprocess_text(decoded_preds, decoded_labels, decoded_inputs)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        # result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result["prediction"] = decoded_preds
        result["labels"] = decoded_labels
        result["inputs"] = decoded_inputs
        return result
    
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        # pad_to_multiple_of=8
    )

    # Hugging Face repository id
    repository_id = f"{model_name.split('/')[1]}-english"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=200,
        generation_num_beams=4,
        fp16=False, # Overflows with fp16
        learning_rate=2e-5,
        num_train_epochs=4,
        include_inputs_for_metrics=True,
        # logging & evaluation strategies
        # logging_dir=f"{repository_id}/logs",
        # logging_strategy="steps",
        # logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        # report_to="tensorboard",
        # push_to_hub=False,
        # hub_strategy="every_save",
        # hub_model_id=repository_id,
        # hub_token=HfFolder.get_token(),
    )

    print("----------------------------------------------------")
    print(train_data[0])
    # print(tokenizer.decode([train_data[0]]))
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("Flan-T5-XL-English-Pretrained-2e-05-4Epochs")