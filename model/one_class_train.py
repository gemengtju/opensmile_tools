# encoding:utf-8

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import time
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def show_result(pos_prob, y_true, x_val_names, version=1):
    pos = y_true[y_true == 1]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    recall = []
    precision = []
    tp = 0
    fp = 0
    with open('top_result_%s.txt'%str(version), 'w') as wf:
        for i in range(len(threshold)):
            wf.write(str(threshold[i]))
            wf.write(' ')
            wf.write(str(y[i]))
            wf.write(' ')
            wf.write(x_val_names[i])
            wf.write('\n')
            if y[i] == 1:
                tp += 1
                recall.append(float(tp) / len(pos))
                precision.append(float(tp) / (tp + fp))
            else:
                fp += 1
                recall.append(float(tp) / len(pos))
                precision.append(float(tp) / (tp + fp))
            #if int(recall[-1]) == 1:
            #    print precision[-1]
            #    sys.exit(1)
    with open('recall_pre_%s.txt'%str(version), 'w') as wf:
        for i, _ in enumerate(recall):
            wf.write(str(recall[i]))
            wf.write(' ')
            wf.write(str(precision[i]))
            wf.write('\n')
    auc = roc_auc_score(y_true, pos_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="linear svm (AUC: {:.3f})".format(auc), linewidth=2)
    plt.ylim(0, 0.01)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision Recall Curve", fontsize=17)
    plt.legend(fontsize=16)
    plt.savefig('pr_%s.pdf'%str(version))


def one_class_svm_core(x_train, x_test, y_test, x_test_names, version=0):
    clf = OneClassSVM()
    print 'svm begin...'
    start = time.time()
    clf.fit(x_train)
    joblib.dump(clf, 'one_class_model.m')
    print 'begin compute', time.time() - start
    y_distance = clf.decision_function(x_test)
    y_score = (np.clip(y_distance, -1, 1) + 1) / 2
    print 'svm complete..'
    show_result(y_score, y_test, x_test_names, version=version)


def one_class_iforest_core(x_train, x_test, y_test, x_test_names, version=0):
    clf = IsolationForest(behaviour='new', max_samples=10000, n_estimators=1000)
    print 'iforest begin...'
    start = time.time()
    clf.fit(x_train)
    joblib.dump(clf, 'one_class_iforest_model.m')
    print 'begin compute', time.time() - start
    y_score = clf.decision_function(x_test)
    y_score = -1 * y_score + 0.5
    print 'iforest complete..'
    show_result(y_score, y_test, x_test_names, version=version)


def train(i):
    data = np.load('train_test_data/one_class_train.npz')
    #data = np.load('train_wav_one0.npz')
    x_train = data['x']
    data = np.load('train_test_data/test_110.npz')
    #data = np.load('test_wav_one0.npz')
    x_test = data['x']
    y_test = data['y']
    x_test_names = data['name']
    #one_class_svm_core(x_train, x_test, y_test, x_test_names, version=i)
    one_class_iforest_core(x_train, x_test, y_test, x_test_names, version=i)


if __name__ == '__main__':
    #second_model_fetch_data(11)
    #compute_core(11)
    train(30)
