import wfdb
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import csv

chunks = []

### Create Stratified Label ###
with open('dataset/label.csv') as fp:
    rdr = csv.reader(fp)
    next(rdr)

    row_count = sum(1 for row in rdr)
    fp.seek(0)
    next(rdr)

    tmplst = []

    for row in rdr:
        tmplst.append(row)

    for i in range(1, 11):
        chunks.append([])
        for j in range(row_count):
            if int(tmplst[j][2]) == i:
                chunks[i - 1].append(tmplst[j])

for i in range(1, 11):
    with open('dataset/strat_fold-' + str(i) + '.csv', 'w') as fp:
        wtr = csv.writer(fp)

        header = ['class', 'fname', 'fold']
        wtr.writerow(header)
        wtr.writerows(chunks[i - 1])

## ANOVA!
### Evaluate using stratified folds ###
for j in range(1, 11, 2):
    tmp_y = pd.concat([pd.read_csv('dataset/strat_fold-' + str(j) + '.csv'), pd.read_csv('dataset/strat_fold-' + str(j + 1) + '.csv')])
    X = np.array([wfdb.rdsamp('dataset/' + f)[0] for f in tmp_y.fname])

    Y = []
    for i in tmp_y[['class']].values:
        # NORM, MI(STEMI), STTC(NSTEMI), CD(부정맥), HYP(비대), ETC
        if i[0] == 0: # 정상(NORM)
            Y.append(0)
        else: # 비정상(ETCs)
            Y.append(1)

    Y = np.array(Y)
    X = tf.cast(X, tf.float32)

    loss, acc = model1.evaluate(X, Y)
    print(f"# ----- #")
    print(f"Model 1. Loss = {loss}, Acc = {acc}")
    loss, acc = model2.evaluate(X, Y)
    print(f"Model 2. Loss = {loss}, Acc = {acc}")

## 모든 데이터에 대해 다시 검증 ##
tmp_y = pd.read_csv('dataset/label.csv')
X = np.array([wfdb.rdsamp('dataset/' + f)[0] for f in tmp_y.fname])
Y = []
for i in tmp_y[['class']].values:
    # NORM, MI(STEMI), STTC(NSTEMI), CD(부정맥), HYP(비대), ETC
    if i[0] == 0: # 정상(NORM)
        Y.append(0)
    else: # 비정상(ETCs)
        Y.append(1)

Y = np.array(Y)
X = tf.cast(X, tf.float32)

model = load_model("pretty-batch128.keras")

y_prediction = model.predict(X)

result = confusion_matrix(Y, np.where(y_prediction > 0.5, 1, 0))
print(result)

fileidx = input("ECG 번호 입력(00000 Format): ") + '_lr'

tmp_y = pd.read_csv('dataset/label.csv') # 01111_lr -> 1, 00010 -> 0
file = tmp_y['fname'].str.contains(fileidx)
subdf = tmp_y[file]

record = wfdb.rdrecord('dataset/' + subdf.iloc[0,1])
wfdb.plot_items(signal=record.p_signal, figsize=(20, 8))

X = np.array([wfdb.rdsamp('dataset/' + f)[0] for f in subdf.fname])
Y = []
for i in tmp_y[['class']].values:
    # NORM, MI(STEMI), STTC(NSTEMI), CD(부정맥), HYP(비대), ETC
    if i[0] == 0: # 정상(NORM)
        Y.append(0)
    else: # 비정상(ETCs)
        Y.append(1)
Y = np.array(Y)
print("NORM" if np.where(model.predict(X) > 0.5, 1, 0)[0][0] == 0 else "ANOM")