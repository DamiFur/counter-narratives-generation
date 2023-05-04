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
import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import argparse

device = torch.device("cuda")
parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument("dataset", type=str, choices=["conan", "asohmo", "both"])
parser.add_argument("generation_strategy", type=str, choices=["zeroshot", "fewshot", "finetuned", "pretraining"])
parser.add_argument("language", type=str, choices=["english", "multi"])
parser.add_argument("--use_extra_info", type=str, choices=["collective", "premises", "all", ""], default="")
parser.add_argument("--model_name", type=str, default="google/flan-t5-base")

args = parser.parse_args()

model_name = args.model_name
language = args.language

FEWSHOT_EXAMPLES_AMOUNT = 2
fewshot_examples = {}

def load_conan(language):

    j = open("dataset/CONAN/CONAN.json", "r")
    conan = json.load(j)
    if language == "english":
        conan_dataset = [{**dct, **{"language": "EN"}} for dct in filter(lambda cn: cn["cn_id"].startswith("EN"), conan["conan"])]
    elif language == "multi":
        conan_dataset_fr = [{**dct, **{"language": "FR"}} for dct in filter(lambda cn: cn["cn_id"].startswith("FR"), conan["conan"])]
        conan_dataset_it = [{**dct, **{"language": "IT"}} for dct in filter(lambda cn: cn["cn_id"].startswith("IT"), conan["conan"])]
        conan_dataset = conan_dataset_fr + conan_dataset_it

    group_by_tweet = {}
    for cn in conan_dataset:
        if not cn["hateSpeech"] in group_by_tweet:
            group_by_tweet[cn["hateSpeech"]] = [[cn["counterSpeech"]], cn["language"]]
        else:
            group_by_tweet[cn["hateSpeech"]][0].append(cn["counterSpeech"])

    acum = 0
    val_threshold = len(conan_dataset) * 0.8
    if args.generation_strategy == "pretraining":
        train_threshold = len(conan_dataset) * 0.7
        train_dataset = []
        val_dataset = []
    test_dataset = []
    keys = list(group_by_tweet.keys())
    keys.sort()
    random.seed(42)
    random.shuffle(keys)
    current_fewshot_examples = {}
    for key in keys:
        if args.generation_strategy == "pretraining":
            if acum < train_threshold:
                train_dataset.append({"hateSpeech": key, "counterSpeech": group_by_tweet[key][0], "language": group_by_tweet[key][1]})
            elif acum < val_threshold:
                val_dataset.append({"hateSpeech": key, "counterSpeech": group_by_tweet[key][0], "language": group_by_tweet[key][1]})
        elif args.generation_strategy == "fewshot":
            language = group_by_tweet[key][1]
            if language not in current_fewshot_examples:
                current_fewshot_examples[language] = 1
                fewshot_examples[language] = [{"hateSpeech": key, "counterSpeech": group_by_tweet[key][0][0]}]
            elif current_fewshot_examples[language] < FEWSHOT_EXAMPLES_AMOUNT:
                current_fewshot_examples[language] += 1
                fewshot_examples[language].append({"hateSpeech": key, "counterSpeech": group_by_tweet[key][0][0]})
        if acum >= val_threshold:
            test_dataset.append({"hateSpeech": key, "counterSpeech": group_by_tweet[key][0], "language": group_by_tweet[key][1]})

        acum += len(group_by_tweet[key][0])
    if args.generation_strategy == "pretraining":
        return [test_data, train_dataset, val_dataset]
    return [test_dataset]

def parse_dataset(filenames, use_extra_info="", language="english"):
    cns_by_tweet = {}
    nonargs = 0
    cn_length = 0
    for filename in glob(filenames):
        f = open(filename, "r")
        tweet_list = []
        is_arg = True
        need_collective = use_extra_info == "collective" or use_extra_info == "all"
        need_premises = use_extra_info == "premises" or use_extra_info == "all"
        if need_collective:
            collective = []
            consecutive_collective = False
            property = []
            consecutive_property = False
        if need_premises:
            justification = []
            consecutive_just = False
            conclusion = []
            consecutive_conc = False
            pivot = []
            consecutive_pivot = False
        prev_line = ["", "", "", "", "", "", "", "", ""]
        for line in f:
            splitted_line = line.split("\t")
            if splitted_line[1].startswith("NoArgumentative"):
                is_arg = False
                break
            if splitted_line[4].startswith("Collective") and need_collective:
                if not prev_line[4].startswith("Collective") and consecutive_collective:
                    collective.append(" - ")
                collective.append(splitted_line[0])
                consecutive_collective = True
            if splitted_line[5].startswith("Property") and need_collective:
                if not prev_line[5].startswith("Property") and consecutive_property:
                    property.append(" - ")
                property.append(splitted_line[0])
                consecutive_property = True
            if splitted_line[2].startswith("Premise2Justification") and need_premises:
                if not prev_line[2].startswith("Premise2Justification") and consecutive_just:
                    justification.append(" - ")
                justification.append(splitted_line[0])
                consecutive_just = True
            if splitted_line[3].startswith("Premise1Conclusion") and need_premises:
                if not prev_line[3].startswith("Premise1Conclusion") and consecutive_conc:
                    conclusion.append(" - ")
                conclusion.append(splitted_line[0])
                consecutive_conc = True
            if splitted_line[6].startswith("pivot") and need_premises:
                if not prev_line[6].startswith("pivot") and consecutive_pivot:
                    pivot.append(" - ")
                pivot.append(splitted_line[0])
                consecutive_pivot = True
            if (not splitted_line[7].startswith("O")) and need_premises:
                type_just = splitted_line[7].strip()
            if (not splitted_line[8].startswith("O")) and need_premises:
                type_conc = splitted_line[8].strip()

            tweet_list.append(splitted_line[0])
            prev_line = splitted_line
            # if splitted_line[]
        if not is_arg:
            nonargs += 1
            continue
        tweet = " ".join(tweet_list)
        if need_collective:
            if language == "english":
                extra_info = " | Collective: " + " ".join(collective) + " | Property: " + " ".join(property)
            else:
                extra_info = " | Colectivo: " + " ".join(collective) + " | Propiedad: " + " ".join(property)
        elif need_premises:
            if language == "english":
                extra_info = " | Justification: " + " ".join(justification) + " (" + type_just + ") " + " | Conclusion: " + " ".join(conclusion) + " (" + type_conc + ") " + " | Pivot: " + " ".join(pivot)
            else:
                extra_info = " | Justificación: " + " ".join(justification) + " (" + type_just + ") " + " | Conclusión: " + " ".join(conclusion) +  " (" + type_conc + ") " + " | Pivot: " + " ".join(pivot)
        else:
            extra_info = ""

        # print(tweet)
        counternarratives = []
        cn = open(filename.replace("conll", "cn"), "r")
        for line in cn:
            counternarratives.append(line)
        if tweet in cns_by_tweet:
            cns_by_tweet[tweet]["cns"] += counternarratives
        else:
            cns_by_tweet[tweet] = {"cns": counternarratives, "lang": "EN" if language == "english" else "ES", "extra_info": extra_info}
        cn_length += len(counternarratives)
    return cns_by_tweet, nonargs, cn_length


def load_asohmo(language, use_extra_info=""):

    if language == "english":
        cns_by_tweet, nonargs, cn_length = parse_dataset("dataset/ASOHMO/english/*.conll", use_extra_info=use_extra_info, language=language)
    elif language == "multi":
        cns_by_tweet, nonargs, cn_length = parse_dataset("dataset/ASOHMO/spanish/*.conll", use_extra_info=use_extra_info, language=language)
        if args.generation_strategy == "pretraining":
            cns_by_tweet2, nonargs2, cn_length2 = parse_dataset("dataset/ASOHMO/english/*.conll", use_extra_info=use_extra_info, language="english")
            cns_by_tweet = {**cns_by_tweet, **cns_by_tweet2}
            nonargs += nonargs2
            cn_length += cn_length2
    print(f"Non arg examples discarted for not having CN: {nonargs}")
    if args.generation_strategy == "pretraining":
        train_threshold = cn_length * 0.7
        train_dataset = []
        val_dataset = []
    val_threshold = cn_length * 0.8
    test_dataset = []
    acum = 0
    keys = list(cns_by_tweet.keys())
    keys.sort()
    random.seed(42)
    random.shuffle(keys)
    current_fewshot_examples = {}
    for key in keys:
        # print(to_append)
        if args.generation_strategy == "pretraining":
            for cn in cns_by_tweet[key]["cns"]:
                to_append = {"hateSpeech": key + cns_by_tweet[key]["extra_info"], "counterSpeech": cn, "language": cns_by_tweet[key]["lang"]}
                if acum < train_threshold:
                    train_dataset.append(to_append)
                elif acum < val_threshold:
                    val_dataset.append(to_append)
                else:
                    test_dataset.append(to_append)
        else:
            to_append = {"hateSpeech": key + cns_by_tweet[key]["extra_info"], "counterSpeech": cns_by_tweet[key]["cns"], "language": cns_by_tweet[key]["lang"]}
            if args.generation_strategy == "fewshot":
                language_code = "ES" if language == "multi" else "EN"
                if language_code not in current_fewshot_examples:
                    current_fewshot_examples[language_code] = 1
                    fewshot_examples[language_code] = [{"hateSpeech": key, "counterSpeech": cns_by_tweet[key]["cns"][0]}]
                elif current_fewshot_examples[language_code] < FEWSHOT_EXAMPLES_AMOUNT:
                    current_fewshot_examples[language_code] += 1
                    fewshot_examples[language_code].append({"hateSpeech": key, "counterSpeech": cns_by_tweet[key]["cns"][0]})
            if acum >= val_threshold:
                test_dataset.append(to_append)
        acum += len(cns_by_tweet[key]["cns"])

    # test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
    if args.generation_strategy == "pretraining":
        return [test_dataset, train_dataset, val_dataset]
    return [test_dataset]

if args.dataset == "conan":
    print("loading conan")  
    datasetss = load_conan(args.language)
elif args.dataset == "asohmo":
    print("loading asohmo")
    datasetss = load_asohmo(args.language, use_extra_info=args.use_extra_info)
else:
    print("loading both datasets")
    datasetss1 = load_asohmo(args.language)
    datasetss2 = load_conan(args.language)
    datasetss = [dtst1 + dtst2 for dtst1, dtst2 in zip(datasetss1, datasetss2)]

test_dataset = datasetss[0]
if args.generation_strategy == "pretraining":
    train_dataset = datasetss[1]
    val_dataset = datasetss[2]

test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
if args.generation_strategy == "pretraining":
    train_data = Dataset.from_pandas(pd.DataFrame(train_dataset))
    val_data = Dataset.from_pandas(pd.DataFrame(val_dataset))

tokenizer = AutoTokenizer.from_pretrained(model_name)

new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]
num_new_tokens = tokenizer.add_tokens(new_tokens)
print("ECN token: ")
print(tokenizer("<ECN>"))

print("We added ", num_new_tokens, " new tokens")

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

def generate_prompt(text, strategy, language):
    initial_prompt = "Generate a non-hateful response for the following hate messages:\n"
    if language == "ES":
        initial_prompt = "Genera una respuesta sin odio a los siguientes mensajes de odio:\n"
    elif language == "FR":
        initial_prompt = "Générez une réponse pour les messages haineux suivants:\n"
    elif language == "IT":
        initial_prompt = "Genera una risposta per i seguenti messaggi di odio:\n"

    if strategy == "zeroshot":
        return  initial_prompt + text + " => "
    elif strategy == "fewshot":
        example_list = []
        for examp in fewshot_examples[language]:
            example_list.append(examp["hateSpeech"] + " => " + examp["counterSpeech"])
        examples = '\n'.join(example_list)
        # examples = """
        #     I hate Muslims. They should not exist. => Muslims are human too. People can choose their own religion.
        #     All immigrants are criminals and should be deported. => Most immigrants are hard-working people trying to have a better life for them and their families.
        #     Immigrants are lazy and cost a lot of money to the taxpayers. => Immigrants usually have the worst jobs and pay taxes as everyone else.
        # """
        # if language == "ES":
            # examples = """
            #     Odio a los musulmanes. No deberían existir. => Los musulmanes también son seres humanos. Las personas pueden elegir su propia religión.
            #     Todos los inmigrantes son criminales y deberían ser deportados. => La mayoría de los inmigrantes son gente trabajadora tratando de tener una vida mejor para ellos y sus familias.
            #     Los inmigrantes son vagos y cuestan mucho dinero a los contribuyentes. => Los inmigrantes usualmente tienen los peores trabajos y pagan impuestos como cualquier otro.
            # """
        # if language == "FR":
            # examples = """
            #     Je déteste les musulmans. Ils ne devraient pas exister. => Les musulmans sont humains aussi. Les gens peuvent choisir leur propre religion.
            #     Tous les immigrés sont des criminels et doivent être expulsés. => La plupart des immigrants sont des travailleurs qui essaient d'améliorer leur vie et celle de leur famille.
            #     Les immigrants sont paresseux et coûtent très cher aux contribuables. => Les immigrés ont généralement les pires emplois et paient des impôts comme tout le monde.
            # """
        # if language == "IT":
            # examples = """
            #     Odio i musulmani. Non dovrebbero esistere. => Anche i musulmani sono umani. Le persone possono scegliere la propria religione.
            #     Tutti gli immigrati sono criminali e dovrebbero essere espulsi. => La maggior parte degli immigrati sono persone laboriose che cercano di avere una vita migliore per loro e per le loro famiglie.
            #     Gli immigrati sono pigri e costano un sacco di soldi ai contribuenti. => Gli immigrati di solito hanno i lavori peggiori e pagano le tasse come tutti gli altri.
            # """
            
        # print("==============================================")
        return initial_prompt + examples + '\n' + text + " => "
    elif strategy == "finetuned" or strategy == "pretraining":
        return initial_prompt + "<SHS>" + text + "<EHS> => "

datasett = test_dataset
if args.generation_strategy == "pretraining":
    datasett += train_dataset
    datasett += val_dataset

dataset_tokenized = list(map(lambda sample: tokenizer(generate_prompt(sample["hateSpeech"], args.generation_strategy, sample["language"]), truncation=True)["input_ids"], datasett))
max_source_length = max([len(x) for x in dataset_tokenized])

if args.generation_strategy == "pretraining":
    target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], datasett))
    max_target_length = max([len(x) for x in target_tokenized])


def preprocess(sample, padding="max_length"):
    inputs = generate_prompt(sample["hateSpeech"], args.generation_strategy, sample["language"])
    model_inputs = tokenizer(inputs, padding=padding, max_length=max_source_length, truncation=True, return_tensors="pt")
    model_inputs = model_inputs.to(device)
    if args.generation_strategy == "pretraining":
        model_inputs["input_ids"] = torch.flatten(model_inputs["input_ids"])
        labels = tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", padding=padding, max_length=max_target_length, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [
                (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
    else:
        model_inputs["labels"] = sample["counterSpeech"]
    return model_inputs

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

    filename = f"{args.dataset}_{args.model_name}_{args.language}_2e-05_{args.generation_strategy}_{args.use_extra_info}_{top_sampling}_{beam_search}_{temperature}".replace("/", "-")
    w = open(filename, 'w')
    for example in testing_datasets:
        inputt = example[0]
        tweet = example[1]
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
        print("----------------------------------tweet-----------------------------")
        print(tweet)
        print("----------------------------------preds-----------------------------")
        print(preds)
        print("\n")
        for labels in inputt["labels"]:

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


if args.generation_strategy == "pretraining":

    train_data = train_data.map(preprocess)
    val_data = val_data.map(preprocess)
    test_data = test_data.map(preprocess)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print(labels)
        print("=============================")
        print(preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        # Some simple post-processing
        # decoded_preds, decoded_labels, decoded_inputs = postprocess_text(decoded_preds, decoded_labels, decoded_inputs)

        # Using rouge score
        result = metric3.compute(predictions=decoded_preds, references=decoded_labels)


        # result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result["prediction"] = decoded_preds
        result["labels"] = decoded_labels
        # result["inputs"] = decoded_inputs
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

    # print(train_data[0])

    # Hugging Face repository id
    repository_id = f"{model_name.split('/')[1]}-english"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        predict_with_generate=True,
        generation_max_length=200,
        generation_num_beams=4,
        # TODO: turn this on and check if it works
        fp16=True, # Overflows with fp16
        learning_rate=2e-4,
        num_train_epochs=8,
        # include_inputs_for_metrics=True,
        # logging & evaluation strategies
        # logging_dir=f"{repository_id}/logs",
        # logging_strategy="steps",
        logging_steps=5,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # save_total_limit=10,
        # load_best_model_at_end=True,
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

    trainer.save_model(f"{args.dataset}_{args.model_name}_{args.language}_{args.use_extra_info}_2e-05_8Epochs".replace("/", "-"))
else:
    preprocessed_dataset = []
    for example in test_data:
        preprocessed_dataset.append([preprocess(example), example["hateSpeech"]])

    print("generating")
    evaluate_generation(preprocessed_dataset)
    evaluate_generation(preprocessed_dataset, top_sampling=True)
    evaluate_generation(preprocessed_dataset, temperature=True)
    evaluate_generation(preprocessed_dataset, beam_search=True)