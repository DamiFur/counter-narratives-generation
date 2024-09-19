import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Add file ID to example tweets in CN generated files")
parser.add_argument("language", type=str, choices=["english", "spanish"], default="spanish")

#TODO: copiar el archivo de español, parametrizar el idioma, hacer la división en train y test de modo que no queden tweets repetidos
args = parser.parse_args()

def find_files_with_string_conll(search_string, folder_path):
    matching_files = []

    # Traverse the directory tree
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is a CoNLL file
            if not file_path.endswith('.conll'):
                continue
            
            # Read the file and join all words in the first column
            try:
                words = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            word = line.split()[0]  # Get the first column (first word)
                            words.append(word)
                
                # Concatenate all words into a single string
                concatenated_text = ' '.join(words)

                # Check if the concatenated text contains the search string
                if search_string in concatenated_text:
                    matching_files.append(file_path)

            except (UnicodeDecodeError, IOError, IndexError):
                # Ignore files that cannot be decoded as text or read, or lines without columns
                pass
    
    return matching_files[0].split("_")[-1].replace(".conll", "")
col_names = ["tweet", "cn", "offensive", "stance", "informativeness", "felicity"]
data = pd.read_csv(f"cn_dataset_{args.language}.csv", names=col_names)
folder_path = f"dataset/ASOHMO/{args.language}/"

data['tweet'] = data['tweet'].apply(lambda x: x.split(" |")[0])
data['cn'] = data['cn'].apply(lambda x: x.replace("<pad>", "").replace("<unk>", "").replace("<s>", "").replace("</s>", "").replace("<SCN>", "").replace("<ECN>", ""))
data['tweet_id'] = data['tweet'].apply(lambda x: find_files_with_string_conll(x.split(" |")[0], folder_path))

# unique_values = list(data['tweet_id'].unique())

# print(unique_values)
data.to_csv(f"cn_dataset_{args.language}_with_id.csv", index=False)



# for example in data.iterrows():
#     tweet_text = example[1]['tweet'].split(" |")[0]
#     result = find_files_with_string_conll(tweet_text, folder_path)
#     print(result)

# search_string = "@user must deport all illegal migrants india already reeling under constant threat of muslim radicals curb population"
# result = find_files_with_string_conll(search_string, folder_path)
# print(result)

