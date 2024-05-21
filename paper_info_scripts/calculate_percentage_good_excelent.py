from glob import glob
for f in glob("./results_automatic_evaluation/*"):
    lines = 0
    goods = 0
    excelent = 0
    for line in open(f):
        lines += 1
        if line.startswith("======================="):
            break
        predsStr = line.split("\t")[2]
        preds = [v.replace("[", "").replace("]", "").replace("tensor(", "").replace(")", "").replace(" ", "") for v in predsStr.split(",")]
        assert(len(preds) == 4)
        if preds[0] == "2" and preds[1] == "2" and preds[3] == "2":
            goods += 1
            if preds[2] == "2":
                excelent += 1
    print("===========================")
    print(f)
    print(goods)
    print(excelent)
    print(lines)