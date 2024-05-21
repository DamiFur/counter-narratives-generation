from glob import glob
import argparse
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
from transformers import Trainer
from transformers import EvalPrediction
from sklearn import metrics
import torch
from statistics import mean

parser = argparse.ArgumentParser(description="Extracts counter-narratives from folder of output files and evaluates them using an automatic model. Exports results in a tsv")
parser.add_argument("model_name", type=str)
# parser.add_argument("category", type=str, choices=["stance", "offensive", "felicity", "informativeness"])
parser.add_argument("language", type=str, choices=["english", "spanish"])
args = parser.parse_args()

model_name = args.model_name
# category = args.category
language = args.language
lr = 2e-05
# This is the order in which they are printed on the output file
categories = ["stance", "offensive", "informativeness", "felicity"]
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

def tokenize_example(example):
    input_text = example["hs"] + " [SEP] " + example["cn"]
    tokenized_input = tokenizer(input_text, truncation=True)
    return tokenized_input

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    all_true_labels = [str(label) for label in labels]
    all_true_preds = [str(pred) for pred in preds]
    avrge = "macro"
    f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None, labels=['0','1','2'])

    f1 = metrics.f1_score(all_true_labels, all_true_preds, average=avrge)

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average=avrge)

    precision = metrics.precision_score(all_true_labels, all_true_preds, average=avrge)

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}-metrics".format(lr, model_name), "a")

    w.write("{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall)))
    w.close()

    ans = {
        'accuracy': acc,
        'f1': f1,
        'f1_per_category': f1_all,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': str(confusion_matrix),
    }

    return ans

tw_cn_pairs = {}
<<<<<<< HEAD
lang = language if language == "english" else "multi"
# ./test_results_kr/asohmo_google-flan-t5-*_{lang}_*

for f in glob(f"./test_results_kr/*_{language}_*"):
    filename_splitted = f.split("_")
    tweets = []
    cns = []
    dont_add = False
    for idx, line in enumerate(open(f, "r")):
        line = line.replace("\n","").replace("\t"," ")
        if line.startswith("===="):
            break
        if idx % 9 == 1:
            if line not in tweets:
                tweet_to_add = line
            else:
                dont_add = True
        elif idx % 9 == 4:
            if line not in cns:
                if not dont_add:
                    tweets.append(tweet_to_add)
                    cns.append(line)
            dont_add = False
    assert(len(tweets) == len(cns))
    tw_cn_pairs[f] = [tweets, cns]



predictions = {}
for category in categories:
    model_name_short = model_name.split("/")[-1].lower()
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name_short}-{category}-{language}-{lr}", num_labels=3)
    avg_score = 0
    l = 0

    for f in tw_cn_pairs:

        if f not in predictions:
            predictions[f] = {"categories": []}

        filename = "./results_test_{}".format(f)

        tweets, cns = tw_cn_pairs[f]
        for tw, cn in zip(tweets, cns):
            only_tw_text = tw.split("|")[0]
            exmpl = only_tw_text + " [SEP] " + cn
            tokenized_input = tokenizer(exmpl, truncation=True, return_tensors="pt")
            output = model.forward(**tokenized_input).logits
            prediction = torch.argmax(output, dim=1)[0]

            if only_tw_text not in predictions[f]:
                predictions[f][only_tw_text] = []

            predictions[f][only_tw_text].append(prediction)
            avg_score += prediction
            l += 1
        predictions[f]['categories'].append(avg_score / l)
            
for f in tw_cn_pairs:
    w = open("results_{}.tsv".format(f.split("/")[-1]), 'w')
    twts, cns = tw_cn_pairs[f]
    scores = []
    for tw, cn in zip(twts, cns):
        only_tw_text = tw.split("|")[0]
        preds = predictions[f][only_tw_text]
        score = int(preds[0]) + int(preds[1]) + int(preds[3]) + 3
        if int(preds[1]) == 3:
            score += int(preds[2]) + 1
        score = score -2
        scores.append(score)
        w.write("{}\t{}\t{}\t{}\n".format(only_tw_text.replace("\t"," "), cn.replace("\t"," "), preds, score))

    w.write("===========================================\n")
    w.write("\t".join([str(float(prd)) for prd in predictions[f]['categories']]))
    w.write("\n\n")
    w.write(str(mean(scores)))
