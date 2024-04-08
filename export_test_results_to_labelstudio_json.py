from glob import glob
import argparse
import os

parser = argparse.ArgumentParser(description="Extracts counter-narratives from folder of output files and evaluates them using an automatic model. Exports results in a tsv")
parser.add_argument("--finetuned", type=bool, default=False)
parser.add_argument("--language", type=str, choices=["english", "spanish"], default="spanish")
parser.add_argument("--argumentative_info", type=str, choices=["", "all", "collective", "premises"], default="")
parser.add_argument("--type_of_cn", type=str, choices=["", "a", "b", "c"], default="")
args = parser.parse_args()

language = args.language

tweet_to_filename = {}

def apply_template(hs, cn):
    return "{\"data\": {\"text\": \"Lea atentamente el siguiente intercambio de tweets:\n\nDado el siguiente discurso de odio:\n\n" + hs.replace('\"', '\\\"') + "\n\nSe responde lo siguiente:\n\n" + cn.replace('\"', '\\\"') + "\"}}"

for file in glob(f'dataset/ASOHMO/{language}/test/*.conll'):
    with open(file, 'r') as f:
        lines = f.readlines()
        noarg = False
        tweet = []
        for line in lines:
            line_split = line.split('\t')
            if len(line_split) < 2 or line_split[1] == 'NoArgumentative':
                noarg = True
                break
            tweet.append(line_split[0])
        tweet_to_filename[' '.join(tweet)] = file.split("/")[-1].replace(".conll", "")


language_alt = "multi" if language == 'spanish' else language
finetuned_str = "finetuned" if args.finetuned else "fewshot"
filename = f"./test_results_kr/asohmo_google-flan-t5-base_{language_alt}_2e-05_{finetuned_str}_{args.argumentative_info}_{args.type_of_cn}_False_True_False"
if not os.path.exists("json_export"):
    os.mkdir("json_export")
export_folder = f"json_export/flan-t5-base_{language}_2e-05_{finetuned_str}_{args.argumentative_info}_{args.type_of_cn}"
os.mkdir(export_folder)
tweets = []
cns = []
dont_add = False
for idx, line in enumerate(open(filename, "r")):
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

for tw, cn in zip(tweets, cns):
    if not tw in tweet_to_filename:
        print(tw)
        print("------------------------")
    filename = tweet_to_filename[tw]
    with open(f"{export_folder}/{filename}.json", "w") as f:
        f.write(apply_template(tw, cn.replace("<pad> ", "").replace("<pad>", "").replace("</s>", "").replace("<s>","")))


