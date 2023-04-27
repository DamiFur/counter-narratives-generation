import json
import random
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from datasets import Dataset
import evaluate
from sentence_transformers import SentenceTransformer, util


model_name = "google/flan-t5-base"

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
val_threshold = len(english_conan) * 0.8
test_dataset = []
keys = list(group_by_tweet.keys())
random.seed(42)
random.shuffle(keys)
for key in keys:
    if acum > val_threshold:
        test_dataset.append({"hateSpeech": key, "counterSpeech": group_by_tweet[key]})

    acum += len(group_by_tweet[key])


test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
tokenizer = AutoTokenizer.from_pretrained(model_name)

new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]

num_new_tokens = tokenizer.add_tokens(new_tokens)

print("We added ", num_new_tokens, " new tokens")

model = AutoModelForSeq2SeqLM.from_pretrained("Flan-T5-base_English_Prompts_2e-05_8Epochs")
model.resize_token_embeddings(len(tokenizer))

dataset_tokenized = list(map(lambda sample: tokenizer("Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => ", truncation=True)["input_ids"], english_conan))
max_source_length = max([len(x) for x in dataset_tokenized])

# target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], english_conan))
# max_target_length = max([len(x) for x in target_tokenized])

def preprocess(sample, padding="max_length"):
    inputs = "Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => "
    # labels = [sample["counterSpeech"]]
    model_inputs = tokenizer(inputs, padding=padding, max_length=max_source_length, truncation=True, return_tensors="pt")
    # labels = tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", padding=padding, max_length=max_target_length, truncation=True)
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    #     ]
    model_inputs["labels"] = sample["counterSpeech"]
    return model_inputs

test_data = test_data.map(preprocess)

testing_datasets = []
for example in test_data.to_list():
    testing_datasets.append(Dataset.from_list([example]))

sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Metric
metric1 = evaluate.load("bertscore")
metric2 = evaluate.load("bleu")
metric3 = evaluate.load("rouge")


def evaluate_generation(testing_datasets, top_sampling=False, beam_search=False, temperature=False):

    f1_avg = 0.0
    bleu_avg = 0.0
    rouge_avg = 0.0
    sbert_avg = 0.0

    w = open(f"flan-t5-base_english_2e-05_finetuned_{top_sampling}_{beam_search}_{temperature}", 'w')
    for dataset in testing_datasets:
        inputt = preprocess(dataset[0])
        print(inputt)
        if beam_search:
            result = model.generate(**inputt, max_new_tokens=256, eos_token_id=32103, no_repeat_ngram_size=4, num_beams=5, early_stopping=True)
        elif top_sampling:
            result = model.generate(**inputt, max_new_tokens=256, eos_token_id=32103, no_repeat_ngram_size=4, do_sample=True, top_k=0, top_p=0.92)
        elif temperature:
            result = model.generate(**inputt, max_new_tokens=256, eos_token_id=32103, no_repeat_ngram_size=4, do_sample=True, temperature=0.7)
        else:
            result = model.generate(**inputt, max_new_tokens=256, no_repeat_ngram_size=4, eos_token_id=32103)

        for labels in dataset[0]["counterSpeech"]:

            tweet = dataset[0]["hateSpeech"]
            preds = str(tokenizer.batch_decode(result)[0])

            result1 = metric1.compute(predictions=[preds], references=[labels], lang="en")
            result2 = metric2.compute(predictions=[preds], references=[labels])
            result3 = metric3.compute(predictions=[preds], references=[labels])

            cosine_scores_preds = sbert.encode([preds], convert_to_tensor=True)
            cosine_scores_labels = sbert.encode([labels], convert_to_tensor=True)

            sbert_score = util.cos_sim(cosine_scores_preds, cosine_scores_labels)

            f1_avg += result1["f1"][0]
            bleu_avg += result2["bleu"]
            rouge_avg += result3["rougeL"]
            sbert_avg += sbert_score[0][0].item()

            w.write("---------------------------------------------------------\n")
            w.write(tweet)
            w.write('\n')
            w.write(labels)
            w.write("\n")
            w.write(preds)
            w.write("\n")
            w.write(str(result1))
            w.write("\n")
            w.write(str(result2))
            w.write("\n")
            w.write(str(result3))
            w.write("\n")
            w.write(str(sbert_score[0][0].item()))
            w.write("\n")

    w.write("========================================\n")
    w.write("F1 AVG:\n")
    w.write(str(f1_avg / len(testing_datasets)))
    w.write("\n")
    w.write("Bleu AVG:\n")
    w.write(str(bleu_avg / len(testing_datasets)))
    w.write("\n")
    w.write("Rouge AVG:\n")
    w.write(str(rouge_avg / len(testing_datasets)))
    w.write("\n")
    w.write("SBERT AVG:\n")
    w.write(str(sbert_avg / len(testing_datasets)))
    w.close()

print("generating")
# evaluate_generation(testing_datasets)
# evaluate_generation(testing_datasets, top_sampling=True)
# evaluate_generation(testing_datasets, temperature=True)
evaluate_generation(testing_datasets, beam_search=True)