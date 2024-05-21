from glob import glob

language = "spanish"

tweets = {}
cns = {}
cns_a = {}
cns_b = {}
cns_c = {}

for partition in ["train", "test", "dev"]:
    if partition not in tweets:
        tweets[partition] = 0
        cns[partition] = 0
        cns_a[partition] = 0
        cns_b[partition] = 0
        cns_c[partition] = 0
    for f in glob(f"dataset/ASOHMO/{language}/{partition}/*.cn"):
        has_cn = False
        for idx, line in enumerate(open(f, 'r')):
            if line.replace("\n","").replace(" ","").replace("\t","").strip() != "":
                if idx == 0:
                    cns_a[partition] += 1
                elif idx == 1:
                    cns_b[partition] += 1
                elif idx == 2:
                    cns_c[partition] += 1
                cns[partition] += 1
                has_cn = True
        if has_cn:
            tweets[partition] += 1
            

print(tweets)
print(cns)
print(cns_a)
print(cns_b)
print(cns_c)