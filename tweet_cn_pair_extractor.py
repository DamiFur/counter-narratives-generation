from glob import glob

for f in glob("./test_results_generated_cn/asohmo_google-flan-t5-xl_english_2e-05_fewshot_*_False_True_False"):
    filename_splitted = f.split("_")
    dataset = filename_splitted[0]
    model = filename_splitted[1]
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
    # line_of_dots_found = False
    # precision_found = True
    # bleu_found = True
    # rouge_found = True
    # sbert_found = True
    # added_tweet = False
    # added_cn = False
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
                # tweets.append(line)
            # print("Tweet:")
            # print(line.replace("\n",""))
        elif idx % 9 == 4:
            if line not in cns:
                if not dont_add:
                    tweets.append(tweet_to_add)
                    cns.append(line)
            dont_add = False
            # print("cn")
            # print(line.replace("\n",""))
    assert(len(tweets) == len(cns))
    w = open("results_{}.tsv".format(f.split("/")[-1]), 'w')
    for tw, cn in zip(tweets[:20], cns[:20]):
        w.write("{}\t{}\n".format(tw.replace("\t"," "), cn.replace("\t"," ")))
    w.close()
