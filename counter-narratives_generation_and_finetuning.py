import json
import random
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import Dataset
import evaluate
from sentence_transformers import SentenceTransformer, util
from glob import glob
import torch
import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse
import os

device = torch.device("cuda")
parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument("dataset", type=str, choices=["conan", "asohmo", "both"])
parser.add_argument("generation_strategy", type=str, choices=["zeroshot", "fewshot", "finetuned", "pretraining"])
parser.add_argument("language", type=str, choices=["english", "multi"])
parser.add_argument("--use_extra_info", type=str, choices=["collective", "premises", "all", ""], default="")
parser.add_argument("--cn_strategy", type=str, default="", choices=["a", "b", "c", ""])
parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
parser.add_argument("--quantized", type=bool, default=False)

args = parser.parse_args()

model_name = args.model_name
language = args.language
pretraining = args.generation_strategy == "pretraining"
is_causallm = "Mistral" in model_name or "Mixtral" in model_name

model_without_user_interface = "tiiuae/falcon" in model_name

FEWSHOT_EXAMPLES_AMOUNT = 10
fewshot_examples = {}

################################## LOAD DATASETS
#TODO: Move to a diferent file

    # Hugging Face repository id
extra_info = args.use_extra_info if args.use_extra_info != "" else "no-info"
cn_strategy = args.cn_strategy if args.cn_strategy != "" else "no-strategy"
repository_id = f"{model_name.split('/')[1]}_{args.language}_{extra_info}_{cn_strategy}"

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
    if pretraining:
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
        if pretraining:
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
    if pretraining:
        return [test_data, train_dataset, val_dataset]
    return [test_dataset]

def parse_dataset(filenames, use_extra_info="", language="english"):
    cns_by_tweet = {}
    nonargs = 0
    cn_length = 0
    cn_type_not_present = 0
    for filename in glob(filenames):
        f = open(filename, "r")
        tweet_list = []
        is_arg = True
        need_collective = use_extra_info == "collective" or use_extra_info == "all" or use_extra_info == "cn_a" or use_extra_info == "cn_b"
        need_premises = use_extra_info == "premises" or use_extra_info == "all" or use_extra_info == "cn_c" or use_extra_info == "cn_a"
        if need_collective:
            collective = []
            consecutive_collective = False
            property = []
            consecutive_property = False
        if need_premises:
            justification = []
            consecutive_just = False
            if need_premises:
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
        extra_info = {}
        if need_collective:
            extra_info["collective"] = " ".join(collective)
            extra_info["property"] = " ".join(property)
        if need_premises:
            extra_info["justification"] = " ".join(justification) + " (" + type_just + ") "
            extra_info["conclusion"] = " ".join(conclusion) + " (" + type_conc + ") "
            extra_info["pivot"] = " ".join(pivot)

        # print(tweet)
        counternarratives = []
        cn = open(filename.replace("conll", "cn"), "r")
        if use_extra_info.startswith("cn_"):
            cn_not_present = False
        for idx, line in enumerate(cn):
            if use_extra_info == "cn_a" or use_extra_info == "cn_a_no_info":
                if idx == 0:
                    if line.replace("\n", "").strip() != "":
                        counternarratives.append(line)
                    else:
                        cn_not_present = True
            elif use_extra_info == "cn_b" or use_extra_info == "cn_b_no_info":
                if idx == 1:
                    if line.replace("\n", "").strip() != "":
                        counternarratives.append(line)
                    else:
                        cn_not_present = True
            elif use_extra_info == "cn_c" or use_extra_info == "cn_c_no_info":
                if idx == 2:
                    if line.replace("\n", "").strip() != "":
                        counternarratives.append(line)
                    else:
                        cn_not_present = True
            else:
                if line.replace("\n", "").strip() != "":
                    counternarratives.append(line)
        if tweet in cns_by_tweet:
            cns_by_tweet[tweet]["cns"] += counternarratives
        else:
            if use_extra_info.startswith("cn_") and cn_not_present:
                cn_type_not_present += 1
            else:
                cns_by_tweet[tweet] = {"cns": counternarratives, "lang": "EN" if language == "english" else "ES", "extra_info": extra_info}
        cn_length += len(counternarratives)
        # TODO: Change for an assert
        if use_extra_info.startswith("cn_") and len(counternarratives) > 1:
            print("ERRRRRRORRRR")
            print(len(counternarratives))
    return cns_by_tweet, nonargs, cn_length, cn_type_not_present


def load_asohmo(language, use_extra_info=""):

    lang_setting = language.replace("multi", "spanish")
    # if language == "english":
    cns_by_tweet, nonargs, cn_length, cn_type_not_present = parse_dataset(f"dataset/ASOHMO/{lang_setting}/test/*.conll", use_extra_info=use_extra_info, language=language)
    if pretraining:
        cns_by_tweet_train, nonargs2, cn_length2, cn_type_not_present2 = parse_dataset(f"dataset/ASOHMO/{lang_setting}/train/*.conll", use_extra_info=use_extra_info, language=language)
        cns_by_tweet_dev, nonargs3, cn_length3, cn_type_not_present3 = parse_dataset(f"dataset/ASOHMO/{lang_setting}/dev/*.conll", use_extra_info=use_extra_info, language=language)
        print(f"{nonargs} - {nonargs2} - {nonargs3}")
        nonargs += nonargs2 + nonargs3
        cn_length += cn_length2 + cn_length3
        cn_type_not_present += cn_type_not_present2 + cn_type_not_present3
        if language == "multi":
            cns_by_tweet_en, nonargs_en, cn_length_en, cn_type_not_present_en = parse_dataset(f"dataset/ASOHMO/english/test/*.conll", use_extra_info=use_extra_info, language=language)
            cns_by_tweet_train2_en, nonargs2_en, cn_length2_en, cn_type_not_present2_en = parse_dataset(f"dataset/ASOHMO/english/train/*.conll", use_extra_info=use_extra_info, language=language)
            cns_by_tweet_dev3_en, nonargs3_en, cn_length3_en, cn_type_not_present3_en = parse_dataset(f"dataset/ASOHMO/english/dev/*.conll", use_extra_info=use_extra_info, language=language)

            cns_by_tweet = {**cns_by_tweet, **cns_by_tweet_en}
            cns_by_tweet_train = {**cns_by_tweet_train, **cns_by_tweet_train2_en}
            cns_by_tweet_dev = {**cns_by_tweet_dev, **cns_by_tweet_dev3_en}

            nonargs += nonargs_en + nonargs2_en + nonargs3_en
            cn_length += cn_length_en + cn_length2_en + cn_length3_en
            cn_type_not_present += cn_type_not_present_en + cn_type_not_present2_en + cn_type_not_present3_en
    # elif language == "multi":
    #     cns_by_tweet, nonargs, cn_length, cn_type_not_present = parse_dataset("dataset/ASOHMO/spanish/test/*.conll", use_extra_info=use_extra_info, language=language)
    #     if pretraining:
    #         cns_by_tweet2, nonargs2, cn_length2, cn_type_not_present2 = parse_dataset("dataset/ASOHMO/english/*.conll", use_extra_info=use_extra_info, language="english")
    #         cns_by_tweet = {**cns_by_tweet, **cns_by_tweet2}
    #         nonargs += nonargs2
    #         cn_length += cn_length2
    #         cn_type_not_present += cn_type_not_present2
    print(f"Counter narratives without the required type of counter-narrative: {cn_type_not_present}")
    print(f"Non arg examples discarted for not having CN: {nonargs}")
    test_dataset = []
    if pretraining:
        print(f"{len(cns_by_tweet.keys())} - {len(cns_by_tweet_train.keys())} - {len(cns_by_tweet_dev.keys())}")
        train_dataset = []
        val_dataset = []
    # acum = 0
    keys = list(cns_by_tweet.keys())
    keys.sort()
    random.seed(42)
    random.shuffle(keys)
    current_fewshot_examples = {}
    for key in keys:
        if pretraining:
            for cn in cns_by_tweet[key]["cns"]:
                to_append = {"hateSpeech": key, "extra_info": cns_by_tweet[key]["extra_info"], "counterSpeech": cn, "language": cns_by_tweet[key]["lang"]}
                test_dataset.append(to_append)
        else:
            to_append = {"hateSpeech": key, "extra_info": cns_by_tweet[key]["extra_info"], "counterSpeech": cns_by_tweet[key]["cns"], "language": cns_by_tweet[key]["lang"]}
            language_code = "ES" if language == "multi" else "EN"
            if args.generation_strategy == "fewshot" and (language_code not in current_fewshot_examples or current_fewshot_examples[language_code] < FEWSHOT_EXAMPLES_AMOUNT):
                if language_code not in current_fewshot_examples:
                    current_fewshot_examples[language_code] = 1
                    fewshot_examples[language_code] = [{"hateSpeech": key, "extra_info": cns_by_tweet[key]["extra_info"], "counterSpeech": cns_by_tweet[key]["cns"][0]}]
                else:
                    current_fewshot_examples[language_code] += 1
                    fewshot_examples[language_code].append({"hateSpeech": key, "extra_info": cns_by_tweet[key]["extra_info"], "counterSpeech": cns_by_tweet[key]["cns"][0]})
            else:
                test_dataset.append(to_append)
        
        # print(to_append)
    if pretraining:
        for key in cns_by_tweet_train:
            for cn in cns_by_tweet_train[key]["cns"]:
                to_append = {"hateSpeech": key, "extra_info": cns_by_tweet_train[key]["extra_info"], "counterSpeech": cn, "language": cns_by_tweet_train[key]["lang"]}
                train_dataset.append(to_append)
        for key in cns_by_tweet_dev:
            for cn in cns_by_tweet_dev[key]["cns"]:
                to_append = {"hateSpeech": key, "extra_info": cns_by_tweet_dev[key]["extra_info"], "counterSpeech": cn, "language": cns_by_tweet_dev[key]["lang"]}
                val_dataset.append(to_append)

    # test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))
    if pretraining:
        return [test_dataset, train_dataset, val_dataset]
    return [test_dataset]


########################################################################################################################################


if args.dataset == "conan":
    print("loading conan")  
    datasetss = load_conan(args.language)
elif args.dataset == "asohmo":
    print("loading asohmo")
    exxtra_info = args.use_extra_info
    if args.cn_strategy == "a":
        if args.use_extra_info == "premises":
            exxtra_info = "cn_a"
        else:
            exxtra_info = "cn_a_no_info"
    elif args.cn_strategy == "b":
        if args.use_extra_info == "collective":
            exxtra_info = "cn_b"
        else:
            exxtra_info = "cn_b_no_info"
    elif args.cn_strategy == "c":
        if args.use_extra_info == "premises":
            exxtra_info = "cn_c"
        else:
            exxtra_info = "cn_c_no_info"
    datasetss = load_asohmo(args.language, use_extra_info=exxtra_info)
else:
    print("loading both datasets")
    datasetss1 = load_asohmo(args.language)
    datasetss2 = load_conan(args.language)
    datasetss = [dtst1 + dtst2 for dtst1, dtst2 in zip(datasetss1, datasetss2)]

test_dataset = datasetss[0]
test_data = Dataset.from_pandas(pd.DataFrame(test_dataset))

if pretraining:
    train_dataset = datasetss[1]
    val_dataset = datasetss[2]
    train_data = Dataset.from_pandas(pd.DataFrame(train_dataset + val_dataset))


tokenizer = AutoTokenizer.from_pretrained(model_name)
if "Mistral" in model_name or "Mixtral" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
)
if 'flan-t5' in model_name or "Mistral" in model_name or "Mixtral" in model_name:
    # new_tokens = ["<SHS>", "<EHS>", "<SCN>", "<ECN>"]
    # num_new_tokens = tokenizer.add_tokens(new_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    # tkn = tokenizer("<ECN>")
    # eos_token_id = tkn["input_ids"][0]

    # print("We added ", num_new_tokens, " new tokens")

if args.generation_strategy == "finetuned":
    # if args.cn_strategy != "":
    model_name = f"pretrained_models/{args.model_name.split('/')[-1]}_multi_{extra_info}_{cn_strategy}"
    print("LOADING MODEL: ", model_name)
    # else:
    #     model_name = f"pretrained_models/{args.dataset}_{args.model_name.replace('/', '-')}_multi_{args.use_extra_info}_2e-05_8Epochs"

if model_name.startswith("bigscience") or model_name.startswith("aleksickx/llama-7b-hf") or model_name.startswith("EleutherAI/gpt-j-6b") or model_name.startswith("tiiuae/falcon-7b") or model_name.startswith("mistralai/Mistral-7B") or "Mixtral" in model_name or "Mistral" in model_name:
    if args.quantized:

        def print_trainable_parameters(model):
            """
            Prints the number of trainable parameters in the model.
            """
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, resume_download=True)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    model.resize_token_embeddings(len(tokenizer))

def generate_prompt(text, strategy, language, extra_info):

    collective = "" if "collective" not in extra_info else extra_info["collective"]
    prop = "" if "property" not in extra_info else extra_info["property"]
    justification = "" if "justification" not in extra_info else extra_info["justification"]
    conclusion = "" if "conclusion" not in extra_info else extra_info["conclusion"]

    # TODO: Change Spanish for the language taken as arg
    initial_prompt = "You are a NGO operator who only speaks Spanish. You are an expert on generation of counter-speech and counter-narratives against hate messages. You are tasked with generating a response to a hate speech tweet in Spanish. The hate speech tweet is the following:\n\n"

    if model_without_user_interface:
        prompt = f"{initial_prompt}"
        if strategy == "fewshot":
            if args.use_extra_info == "":
                prompt += "'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n"
            if args.use_extra_info == "collective":
                prompt += "'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo' -> Respuesta: 'No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio.'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion' -> Respuesta: 'Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?'\n"
            if args.use_extra_info == "premises":
                prompt += "'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?' -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'  -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n"
            if args.use_extra_info == "all":
                prompt += "'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?' -> Respuesta: 'No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio.'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion'  | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde' -> Respuesta: 'Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?'\n"
        if args.use_extra_info == "":
            prompt += f"{text} -> Respuesta: "
        elif args.use_extra_info == "collective":
            prompt += f"'{text}' | Collectivo contra quien se dirige el odio: '{collective}' | Propiedad asociada al colectivo: '{prop}' -> Respuesta:"
        elif args.use_extra_info == "premises":
            prompt += f"'{text}' | Justificación del argumento: '{justification}' | Conclusión: '{conclusion}' -> Respuesta:"
        elif args.use_extra_info == "all":
            prompt += f"'{text}' | Collectivo contra quien se dirige el odio: '{collective}' | Propiedad asociada al colectivo: '{prop}' | Justificación del argumento: '{justification}' | Conclusión: '{conclusion}' -> Respuesta:"
    else:
        if strategy == "fewshot":
            if args.use_extra_info == "":
                #TODO: Put examples into a cont variable
                prompt = [
                    {"role": "user", "content": f"{initial_prompt}'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings'"},
                    {"role": "assistant", "content": "Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado..."},
                    {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde"},
                    {"role": "assistant", "content": "Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin"},
                    {"role": "user", "content": f"{text}"}
                ]
            elif args.use_extra_info == "collective":
                prompt = [
                    {"role": "user", "content": f"{initial_prompt}'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo'"},
                    {"role": "assistant", "content": "No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio."},
                    {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion'"},
                    {"role": "assistant", "content": "Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?"},
                    {"role": "user", "content": f"{text} | Collectivo contra quien se dirige el odio: '{collective}' | Propiedad asociada al colectivo: '{prop}'"}
                ]
            elif args.use_extra_info == "premises":
                prompt = [
                    {"role": "user", "content": f"{initial_prompt}'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?"},
                    {"role": "assistant", "content": "Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado..."},
                    {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'"},
                    {"role": "assistant", "content": "Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin"},
                    {"role": "user", "content": f"{text} | Justificación del argumento: {justification} | Conclusión: {conclusion}"}
                ]
            elif args.use_extra_info == "all":
                prompt = [
                    {"role": "user", "content": f"{initial_prompt}'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?"},
                    {"role": "assistant", "content": "No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio."},
                    {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'"},
                    {"role": "assistant", "content": "Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?"},
                    {"role": "user", "content": f"{text} | Collectivo contra quien se dirige el odio: '{collective}' | Propiedad asociada al colectivo: '{prop}' | Justificación del argumento: '{justification}' | Conclusión: '{conclusion}'"}
                ]
        else:
            user_prompt = f"{initial_prompt}{text}"
            if args.use_extra_info == "collective":
                user_prompt += f" | Collective: {collective} | Property: {prop}"
            elif args.use_extra_info == "premises":
                user_prompt += f" | Justification: {justification} | Conclusion: {conclusion}"
            elif args.use_extra_info == "all":
                user_prompt += f" | Collective: {collective} | Property: {prop} | Justification: {justification} | Conclusion: {conclusion}"
            prompt = [{"role": "user", "content": user_prompt}]

        return prompt


datasett = test_dataset
if pretraining:
    datasett += train_dataset
    datasett += val_dataset

print(len(test_dataset))
# dataset_tokenized = list(map(lambda sample: tokenizer(generate_prompt(sample["hateSpeech"], args.generation_strategy, sample["language"]), truncation=True)["input_ids"], datasett))
# max_source_length = max([len(x) for x in dataset_tokenized])

# if pretraining:
    # target_tokenized = list(map(lambda sample: tokenizer("<SCN>" + sample["counterSpeech"] + "<ECN>", truncation=True)["input_ids"], datasett))
    # max_target_length = max([len(x) for x in target_tokenized])

MAX_LENGTH = 1024
def preprocess(sample, padding="max_length", is_testing = False):
    inputs = generate_prompt(sample["hateSpeech"], args.generation_strategy, sample["language"], sample["extra_info"])
    print(inputs)
    if pretraining:
        if is_causallm:
            if model_without_user_interface:
                model_inputs = tokenizer(inputs + "<SCN> " + sample["counterSpeech"] + " <ECN>", padding=padding, max_length=MAX_LENGTH, truncation=True)
            else:
                inputs.append({"role": "assistant", "content": sample["counterSpeech"]})
                model_inputs = tokenizer.apply_chat_template(inputs)
            model_inputs["labels"] = model_inputs["input_ids"].copy()

        else:
            model_inputs = tokenizer(inputs, padding=padding, max_length=MAX_LENGTH, truncation=True)
            labels = tokenizer("<SCN> " + sample["counterSpeech"] + " <ECN>", padding=padding, max_length=MAX_LENGTH, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [
                    (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
    else:
        if model_without_user_interface:
            model_inputs = tokenizer(inputs, padding=padding, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
            model_inputs = model_inputs.to(device)
        else:
            model_inputs = tokenizer.apply_chat_template(inputs, return_tensors="pt")
            model_inputs = model_inputs.to(device)
    
    if is_testing:
        model_inputs = {"example": model_inputs, "counterSpeech": sample["counterSpeech"]}
    return model_inputs

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False

if model_name == "tiiuae/falcon-7b-instruct":
    stop_words = [".", "]", "']", "']\n", "\n", "]\n", "\n\n", "']\n\n", "<|endoftext|>"]
else:
    stop_words = [".", "]", "']", "']\n", "\n", "]\n", "\n\n", "']\n\n", "</s>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


def evaluate_generation(testing_datasets, top_sampling=False, beam_search=True, temperature=False):

    # sbert_avg = 0.0

    # sbert = SentenceTransformer('all-MiniLM-L6-v2')

    filename = f"{args.dataset}_{args.model_name}_{args.language}_2e-05_{args.generation_strategy}_{args.use_extra_info}_{args.cn_strategy}_{top_sampling}_{beam_search}_{temperature}".replace("/", "-")
    w = open(filename, 'w')
    for example in testing_datasets:
        inputt = example[0]["example"]
        tweet = example[1]
        # inputt.to(device)
        if beam_search:
            result = model.generate(inputs=inputt, do_sample=True num_beams=4, no_repeat_ngram_size=2, num_return_sequences=1, stopping_criteria=stopping_criteria, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        elif top_sampling:
            result = model.generate(inputs=inputt, do_sample=True top_k=0, top_p=0.92, no_repeat_ngram_size=2, num_return_sequences=1, stopping_criteria=stopping_criteria, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        elif temperature:
            result = model.generate(inputs=inputt, do_sample=True temperature=0.7, no_repeat_ngram_size=2, num_return_sequences=1, stopping_criteria=stopping_criteria, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        else:
            result = model.generate(inputs=inputt no_repeat_ngram_size=2, num_return_sequences=1, stopping_criteria=stopping_criteria, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        preds = str(tokenizer.batch_decode(result)[0])
        print("----------------------------------tweet-----------------------------")
        print(tweet)
        print("----------------------------------preds-----------------------------")
        print(preds)
        print("\n")
        for labels in example[0]["counterSpeech"]:

            # cosine_scores_preds = sbert.encode([preds], convert_to_tensor=True)
            # cosine_scores_labels = sbert.encode([labels], convert_to_tensor=True)

            # sbert_score = util.cos_sim(cosine_scores_preds, cosine_scores_labels)

            # sbert_avg += sbert_score[0][0].item()

            w.write("---------------------------------------------------------\n")
            w.write(tweet)
            w.write('\n')
            w.write(labels)
            w.write("\n")
            w.write(preds)
            w.write("\n")
            # w.write(str(sbert_score[0][0].item()))
            # w.write("\n")

    w.write("========================================\n")
    # w.write("SBERT AVG:\n")
    # w.write(str(sbert_avg / len(testing_datasets)))
    w.close()



if pretraining:
    train_data = train_data.map(preprocess)
    test_data = test_data.map(preprocess)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        # Some simple post-processing
        # decoded_preds, decoded_labels, decoded_inputs = postprocess_text(decoded_preds, decoded_labels, decoded_inputs)
        metric = evaluate.load("bertscore")
        # Using rouge score
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)


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
    )

    # Define training args
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        # predict_with_generate=True,
        # generation_max_length=200,
        # generation_num_beams=4,
        # TODO: turn this on and check if it works
        fp16=False, # Overflows with fp16
        learning_rate=2e-04,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        # include_inputs_for_metrics=True,
        # logging & evaluation strategies
        # logging_dir=f"{repository_id}/logs",
        # logging_strategy="steps",
        # logging_steps=5,
        optim="adamw_bnb_8bit",
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # save_total_limit=10,
        # load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        # report_to="tensorboard",
        # push_to_hub=True,
        # hub_strategy="every_save",
        # hub_model_id=repository_id,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        # eval_dataset=val_data,
        # compute_metrics=compute_metrics,
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")
    model.save_pretrained("pretrained_models/" + repository_id)

    preprocessed_dataset = []
    for example in test_data:
        preprocessed_dataset.append([preprocess(example, is_testing = True), example["hateSpeech"]])

    print("generating")
    # evaluate_generation(preprocessed_dataset)
    # evaluate_generation(preprocessed_dataset, top_sampling=True)
    # evaluate_generation(preprocessed_dataset, temperature=True)
    evaluate_generation(preprocessed_dataset, beam_search=True)
    
    model.push_to_hub("CounterNarratives/" + repository_id)

else:
    preprocessed_dataset = []
    for example in test_data:
        preprocessed_dataset.append([preprocess(example, is_testing = True), example["hateSpeech"]])

    print("generating")
    # evaluate_generation(preprocessed_dataset)
    # evaluate_generation(preprocessed_dataset, top_sampling=True)
    # evaluate_generation(preprocessed_dataset, temperature=True)
    evaluate_generation(preprocessed_dataset, beam_search=True)