import json
import random
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from datasets import Dataset
import evaluate
from sentence_transformers import SentenceTransformer, util
from glob import glob
import torch

import argparse

device = torch.device("cuda")
parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument("dataset", type=str, choices=["conan", "asohmo", "both"])
# parser.add_argument("model", type=str, choices=["zeroshot", "fewshot", "finetuned"])

args = parser.parse_args()


model_name = "google/flan-t5-xl"

def load_conan():

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
    return test_dataset

def load_asohmo():
    cns_by_tweet = {}
    dataset = []
    nonargs = 0
    cn_length = 0
    for filename in glob("dataset/ASOHMO/english/*.conll"):
        # print(filename)
        f = open(filename, "r")
        tweet_list = []
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

    val_threshold = cn_length * 0.8
    test_dataset = []
    acum = 0
    keys = list(cns_by_tweet.keys())
    random.seed(42)
    random.shuffle(keys)
    for key in keys:
        to_append = {"hateSpeech": key, "counterSpeech": cns_by_tweet[key]}
        if acum > val_threshold:
            test_dataset.append(to_append)
        acum += len(cns_by_tweet[key])

    # test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
    return test_dataset

if args.dataset == "conan":   
    test_dataset = load_conan()
else:
    print("loading asohmo")
    test_dataset = load_asohmo()
test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
tokenizer = AutoTokenizer.from_pretrained(model_name)

new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]
num_new_tokens = tokenizer.add_tokens(new_tokens)
print("ECN token: ")
print(tokenizer("<ECN>"))

print("We added ", num_new_tokens, " new tokens")

dataset_tokenized = list(map(lambda sample: tokenizer("Generate a counter-narrative for the following hate speech:\n<SHS>" + sample["hateSpeech"] + "<EHS> => ", truncation=True)["input_ids"], test_dataset))
max_source_length = max([len(x) for x in dataset_tokenized])

# target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], english_conan))
# max_target_length = max([len(x) for x in target_tokenized])

def preprocess(sample, padding="max_length"):
    inputs = "Generate a non-hateful response for the following hate messages:\nI hate Muslims. They should not exist. => Muslims are human too. People can choose their own religion.\nAll immigrants are criminals and should be deported. => Most immigrants are hard-working people trying to have a better life for them and their families.\n Immigrants are lazy and cost a lot of money to the taxpayers. => Immigrants usually have the worst jobs and pay taxes as everyone else.\n" + sample["hateSpeech"] + " => "
    # labels = [sample["counterSpeech"]]
    model_inputs = tokenizer(inputs, padding=padding, max_length=max_source_length, truncation=True, return_tensors="pt")
    # labels = tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", padding=padding, max_length=max_target_length, truncation=True)
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    #     ]
    model_inputs = model_inputs.to(device)
    model_inputs["labels"] = sample["counterSpeech"]
    # model_inputs["hateSpeech"] = inputs
    # model_inputs = model_inputs.to(device)
    return model_inputs

preprocessed_dataset = []
for example in test_data:
    preprocessed_dataset.append([preprocess(example), example["hateSpeech"]])

# print(preprocessed_dataset[0])
# for example in preprocessed_dataset:
#     print(example)
#     dset = Dataset.from_list([example])
#     print(dset[0])
#     dset.to(device)
#     testing_datasets.append(dset)

# print("lllllllllllllllllllllllllllllllllllllll")
# print(testing_datasets[0][0])
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Metric
metric1 = evaluate.load("bertscore")
metric2 = evaluate.load("bleu")
metric3 = evaluate.load("rouge")


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

def evaluate_generation(testing_datasets, top_sampling=False, beam_search=False, temperature=False):

    f1_avg = 0.0
    bleu_avg = 0.0
    rouge_avg = 0.0
    sbert_avg = 0.0

    w = open(f"{args.dataset}_flan-t5-xl_english_2e-05_zeroshot_{top_sampling}_{beam_search}_{temperature}", 'w')
    for example in testing_datasets:
        inputt = example[0]
        tweet = example[1]
        # del inputt["hateSpeech"]
        # del inputt["counterSpeech"]
        # del inputt["labels"]
        # print(inputt)
        # inputt.to(device)
        if beam_search:
            result = model.generate(**inputt, max_new_tokens=512, no_repeat_ngram_size=4, num_beams=5, early_stopping=True)
        elif top_sampling:
            result = model.generate(**inputt, max_new_tokens=512, no_repeat_ngram_size=4, do_sample=True, top_k=0, top_p=0.92)
        elif temperature:
            result = model.generate(**inputt, max_new_tokens=512, no_repeat_ngram_size=4, do_sample=True, temperature=0.7)
        else:
            result = model.generate(**inputt, max_new_tokens=512, no_repeat_ngram_size=4)
        preds = str(tokenizer.batch_decode(result)[0])
        print("----------------------------------preds-----------------------------")
        print(preds)
        print("-----------------------------------tweet-------------------------")
        print(tweet)

        for labels in inputt["labels"]:

            # print("labels:")
            # print(labels)

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
evaluate_generation(preprocessed_dataset)
evaluate_generation(preprocessed_dataset, top_sampling=True)
evaluate_generation(preprocessed_dataset, temperature=True)
evaluate_generation(preprocessed_dataset, beam_search=True)