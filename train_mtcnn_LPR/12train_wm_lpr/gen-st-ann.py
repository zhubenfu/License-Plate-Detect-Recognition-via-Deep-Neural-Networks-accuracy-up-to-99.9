import sys
import os


def readroi(f):
    with open(f) as of:
        s = of.read().split("\n")[1]
        p = s.rfind(",")
        p = s.rfind(",", 0, p-1)
        return s[:p].replace(",", " ")

def listFile(dr, filter):
    fs = os.listdir(dr)
    for i in range(len(fs)-1, -1, -1):
        if not fs[i].lower().endswith(filter):
            del fs[i]
    return fs

root = "samples"
fs = listFile(root, ".jpg")
with open("label.txt", "w") as of:
    for i in range(len(fs)):
        jpg = root + "/" + fs[i]
        roi = readroi(root + "/" + fs[i][:-4] + ".xml.txt")
        line = jpg + " " + roi + "\n"
        of.write(line)


