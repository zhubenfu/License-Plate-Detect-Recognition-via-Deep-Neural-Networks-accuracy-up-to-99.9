import sys
import os

size = "48"
save_dir = "./" + size
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

has_pts = True

f1 = open(os.path.join(save_dir, 'pos_' + size + '.txt'), 'r')
f2 = open(os.path.join(save_dir, 'neg_' + size + '.txt'), 'r')
f3 = open(os.path.join(save_dir, 'part_' + size + '.txt'), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
f = open(os.path.join(save_dir, 'label-train.txt'), 'w')

for i in range(len(pos)):
    p = pos[i].find(" ") + 1
    pos[i] = pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    f.write(pos[i])

for i in range(len(neg)):
    p = neg[i].find(" ") + 1
    neg[i] = neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1" + (" -1 -1 -1 -1 -1 -1 -1 -1" if has_pts else "") + "\n"
    f.write(neg[i])

for i in range(len(part)):
    p = part[i].find(" ") + 1
    part[i] = part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    f.write(part[i])

f1.close()
f2.close()
f3.close()
