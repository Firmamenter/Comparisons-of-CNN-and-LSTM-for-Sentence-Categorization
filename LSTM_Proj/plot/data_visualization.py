#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import os

files = os.listdir("results")
print(files)

datas = []
names = []
for file in files:
    if file.endswith('acc.pkl'):
        with open('results/'+file, 'rb') as f:
            data = pickle.load(f)
            datas.append(data)
            names.append(file)

print(len(datas), len(datas[0]))
for index, name in enumerate(names):
    print(index, file)
    plt.plot([i*1.0/50 for i in range(len(datas[index]))],datas[index], label = name[:-4])
# plt.plot([i*1.0/50 for i in range(len(datas[0]))],datas[0], )
# plt.plot([i*1.0/50 for i in range(len(datas[0]))],datas[1], )
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
