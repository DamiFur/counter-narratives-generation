import random
from glob import glob
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import random_split
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
nltk.download("punkt")

model_name = "google/flan-t5-base"

amount_empty = 0
cns_by_tweet = {}
dataset = []
nonargs = 0
cn_length = 0
for filename in glob("dataset/ASOHMO/english/*.conll"):
    # print(filename)
    f = open(filename, "r")
    tweet_list = []
    all_lines = []
    is_arg = True
    for line in f:
        splitted_line = line.split("\t")
        if splitted_line[1].startswith("NoArgumentative"):
            is_arg = False
            break
        tweet_list.append(line.split("\t")[0])
    if not is_arg:
        nonargs += 1
        continue
    tweet = " ".join(tweet_list)
    # print(tweet)
    counternarratives = []
    cn = open(filename.replace("conll", "cn"), "r")
    for line in cn:
        counternarratives.append(line)
        dataset.append({"hateSpeech": tweet, "counterSpeech": line})
    if tweet in cns_by_tweet:
        cns_by_tweet[tweet] += counternarratives
    else:
        cns_by_tweet[tweet] = counternarratives
    cn_length += len(counternarratives)

train_threshold = cn_length * 0.7
val_threshold = cn_length * 0.8
train_dataset = []
val_dataset = []
test_dataset = []
acum = 0
keys = list(cns_by_tweet.keys())
random.seed(42)
random.shuffle(keys)
for key in keys:
    for cn in cns_by_tweet[key]:
        to_append = {"hateSpeech": key, "counterSpeech": cn}
        if acum < train_threshold:
            train_dataset.append(to_append)
        elif acum < val_threshold:
            val_dataset.append(to_append)
        else:
            test_dataset.append(to_append)
    acum += len(cns_by_tweet[key])

train_data = Dataset.from_pandas(pd.DataFrame(train_dataset))
val_data = Dataset.from_pandas(pd.DataFrame(val_dataset))
test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))

    # print(counternarratives)

tokenizer = AutoTokenizer.from_pretrained(model_name)

new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]

num_new_tokens = tokenizer.add_tokens(new_tokens)

print("We added ", num_new_tokens, " new tokens")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

dataset_tokenized = list(map(lambda sample: tokenizer("Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => ", truncation=True)["input_ids"], dataset))
max_source_length = max([len(x) for x in dataset_tokenized])

target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], dataset))
max_target_length = max([len(x) for x in target_tokenized])

def preprocess(sample, padding="max_length"):
    inputs = "Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => "
    # labels = [sample["counterSpeech"]]
    model_inputs = tokenizer(inputs, padding=padding, max_length=max_source_length, truncation=True)
    labels = tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", padding=padding, max_length=max_target_length, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# random.Random(42).shuffle(english_conan)
train_data = train_data.map(preprocess)
val_data = val_data.map(preprocess)
test_data = test_data.map(preprocess)

print(len(dataset))
print(len(train_data))
print(len(val_data))
print(len(test_data))

# Metric
metric = evaluate.load("rouge")

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

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)


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

print(train_data[0])

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
    num_train_epochs=8,
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

trainer.save_model("ASOHMO_Flan-T5-base_English_Prompts_2e-05_8Epochs")