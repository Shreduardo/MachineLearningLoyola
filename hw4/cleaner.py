import re
import pandas as pd
import sys
sys.path.append("home/Documents/Dev/School/Machine_Learning/hw4")

with open("stop_words.txt", "r") as f:
    stop = f.read()
    f.close()

pos = []
with open("reviews_pos.txt", "r") as f:
    for line in f:
        pos.append(line)
    f.close()

neg = []
with open("reviews_neg.txt", "r") as f:
    for line in f:
        neg.append(line)
    f.close()


posStr = ''
for line in pos:
    posStr += " ".join([word for word in line.split(" ") if word not in stop]) + "\n"

negStr = ''
for line in neg:
    negStr += ' '.join([word for word in line.split(" ") if word not in stop]) + "\n"


with open("PosNoStop.txt", "w") as f:
    f.write(posStr)
    f.close

with open("NegNoStop.txt", "w") as f:
    f.write(negStr)
    f.close()
