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

parser = argparse.ArgumentParser(description="Extracts counter-narratives from folder of output files and evaluates them using an automatic model. Exports results in a tsv")
parser.add_argument("model_name", type=str, choices=["roberta-base", "roberta-large"])
parser.add_argument("category", type=str, choices=["stance", "offensive", "felicity", "informativeness"])
parser.add_argument("language", type=str, choices=["english", "spanish"])
args = parser.parse_args()

model_name = args.model_name
category = args.category
language = args.language
lr = 2e-05
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}-{category}-{language}-{lr}", num_labels=3)

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


for f in glob("./test_results_generated_cn/asohmo_google-flan-t5-xl_english_2e-05_fewshot_*_False_True_False"):
    filename_splitted = f.split("_")
    dataset = filename_splitted[0]
    model_generation = filename_splitted[1]
    language = filename_splitted[2]
    lr = filename_splitted[3]
    strategy = filename_splitted[4]
    extra_info = filename_splitted[5]
    cn_strategy = filename_splitted[6]
    top_sampling = filename_splitted[7]
    beam_search = filename_splitted[8]
    temperature = filename_splitted[9]

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
    # data = pd.DataFrame(list(zip(tweets, cns)), columns=["hs", "cn"])
    # test_set = Dataset.from_pandas(data).map(tokenize_example)
    # trainer = Trainer(
    #     model=model,
    #     eval_dataset=test_set,
    #     tokenizer=tokenizer,
    #     compute_metrics= compute_metrics_f1,
    # )

    # results = trainer.predict(test_set)

    model_name_adapted = model_name.replace("/", "-")
    filename = "./results_test_{}_{}_{}_{}_{}".format(lr, model_generation, model_name_adapted, category, language)

    w = open("results_{}.tsv".format(f.split("/")[-1]), 'w')
    for tw, cn in zip(tweets, cns):
        exmpl = tw + " [SEP] " + cn
        tokenized_input = tokenizer(exmpl, truncation=True, return_tensors="pt")
        output = model.forward(**tokenized_input).logits
        print(exmpl)
        print(torch.argmax(output, dim=1))
        w.write("{}\t{}\t{}\n".format(tw.replace("\t"," "), cn.replace("\t"," ")))
    w.close()
