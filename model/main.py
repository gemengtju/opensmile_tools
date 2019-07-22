# encoding:utf-8

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import linear_model
import time
from collections import defaultdict
import os

# IF path is not exist, create the path
def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def show_result(pos_prob, y_true, x_val_names, version=1):
    pos = y_true[y_true == 1]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    x_val_names = x_val_names[pos_prob.argsort()[::-1]]
    recall = []
    precision = []
    tp = 3
    fp = 5
    
    yuyilist = [('TlRjMk5UTXpNekl3T0RrMk1qSXpNakEz_2', 0.06572970179685922, 0), ('06479', 0.060741379070325674, 1), ('TWpnNE5URTFNVEF6TVRrM09ESTRORGsz_1', 0.05693279992838174, 0), ('TWpnNE5URTBOamcyTWprNE5UZzVOVFE0_1', 0.03412170634291234, 0), ('02956', 0.03298940911013881, 1), ('03410', 0.03264531960304162, 1), ('TlRjMk5UTXpOamMzTVRNM05qRTBORFUz_1', 0.013598570564217538, 0), ('TWpnNE5URTFOREl4TURnMU1qUTRNVGc1_1', 0.010465320902818043, 0)]
    yuyiindex = 0
    with open('./workspace/exp/top_result_%s.txt'%str(version), 'w') as wf:
        for i in range(len(threshold)):
            wf.write(str(threshold[i]))
            wf.write(' ')
            wf.write(str(y[i]))
            wf.write(' ')
            wf.write(x_val_names[i])
            wf.write('\n')
            if yuyiindex < len(yuyilist):
                while threshold[i] <= yuyilist[yuyiindex][1]:
                    if yuyilist[yuyiindex][2] == 0:
                        fp -= 1
                    else:
                        tp -= 1
                    yuyiindex += 1
                    if yuyiindex >= len(yuyilist):
                        break
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
    with open('./workspace/exp/recall_pre_%s.txt'%str(version), 'w') as wf:
        for i, _ in enumerate(recall):
            wf.write(str(recall[i]))
            wf.write(' ')
            wf.write(str(precision[i]))
            wf.write('\n')
    auc = roc_auc_score(y_true, pos_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="linear svm (AUC: {:.3f})".format(auc), linewidth=2)
    plt.ylim(0, 0.3)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision Recall Curve", fontsize=17)
    plt.legend(fontsize=16)
    plt.savefig('./workspace/exp/pr_%s.pdf'%str(version))


def svm_sgd(x_train, y_train, version=1):
    clf = linear_model.SGDClassifier(max_iter=20, loss='modified_huber', early_stopping=True)
    start = time.time()
    print 'svm begin fit', version
    #scaler = preprocessing.StandardScaler().fit(x_train)
    #x_train = scaler.transform(x_train)
    clf.fit(x_train, y_train)
    print 'svm fit completed...', time.time() - start
    makedir("./workspace/exp/")
    joblib.dump(clf, "./workspace/exp/model_wav_%s.m"%str(version))
    #joblib.dump(scaler, "train_scaler_%s.m"%str(version))


def model_test(x_val, y_val, x_val_names, version=1):
    clf = joblib.load('./workspace/exp/model_wav_%s.m'%str(version))
    #clf = joblib.load('model_wav_repaired.m')
    #clf = joblib.load('model_wav_new_vad_second.m')
    #clf = joblib.load('model_wav_new_vad_first.m')
    #scaler = joblib.load('train_scaler.m')
    #x_val = scaler.transform(x_val)
    y_scores = clf.predict_proba(x_val)
    y_scores = y_scores[:,1]
    show_result(y_scores, y_val, x_val_names, version=version)
    return y_scores


def compute_core(i):
    #data = np.load('train_test_data/repaired_train_segment0.npz')
    #data = np.load('train_test_data/repaired_old_train_segment0.npz')
    #data = np.load('train_test_data/repaired_second_train.npz')
    #data = np.load('train_test_data/train_newvad_segment0.npz')
    data = np.load('./workspace/package_feature/train_test_data/20181105_train_segment0.npz')
    #data = np.load('train_test_data/repaired_aaa.npz')
    x_train = data['x']
    y_train = data['y']
    svm_sgd(x_train, y_train, version=i)
    #data = np.load('train_test_data/repaired_test_segment0.npz')
    #data = np.load('train_test_data/repaired_old_test_segment0.npz')
    #data = np.load('train_test_data/test_second_newvad_segment0.npz')
    #data = np.load('train_test_data/20181105_old_wav_train_segment0.npz')
    data = np.load('./workspace/package_feature/train_test_data/20181105_test_segment0.npz')
    #data = np.load('train_test_data/repaired_bbb.npz')
    x_val = data['x']
    y_val = data['y']
    x_val_names = data['name']
    model_test(x_val, y_val, x_val_names, version=i)


def feats_fetch_core(y_scores):
    results = []
    results.append(float(np.min(y_scores, axis=0)[0]))
    results.append(float(np.max(y_scores, axis=0)[0]))
    results.append(float(np.min(y_scores, axis=0)[1]))
    results.append(float(np.max(y_scores, axis=0)[1]))
    # 概率均值
    results.append(float(np.average(y_scores, axis=0)[0]))
    results.append(float(np.average(y_scores, axis=0)[1]))
    # 分位数占比
    data_len = len(y_scores[:, 0])
    results.append(float(len(np.where(y_scores[:, 0] > 0.9))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.9))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.8))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.8))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.7))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.7))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.6))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.6))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.5))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.5))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.4))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.4))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.3))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.3))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.2))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.2))) / data_len)
    results.append(float(len(np.where(y_scores[:, 0] > 0.1))) / data_len)
    results.append(float(len(np.where(y_scores[:, 1] > 0.1))) / data_len)
    return results


def second_model_fetch_data(version):
    clf = joblib.load('train_model_11.m')
    data = np.load('train_test_data/secondtrain_%s0.npz'%str(version))
    x_train = data['x']
    y_scores = clf.predict_proba(x_train)
    x_train_names = data['name']
    y_train = data['y']
    names = defaultdict(list)
    vals = defaultdict(int)
    for i, name in enumerate(x_train_names):
        names[name].append(y_scores[i])
        vals[name] = int(y_train[i])
    x = []
    y = []
    ns = []
    for name in names:
        feats = feats_fetch_core(np.array(names[name]))
        x.append(feats)
        y.append(vals[name])
        ns.append(name)
    np.savez('train_test_data/secondtrain_train_%s'%str(version), x=np.array(x), y=np.array(y), name=np.array(ns))

    data = np.load('train_test_data/test_%s0.npz'%str(version))
    x_train = data['x']
    y_scores = clf.predict_proba(x_train)
    x_train_names = data['name']
    y_train = data['y']
    names = defaultdict(list)
    vals = defaultdict(int)
    for i, name in enumerate(x_train_names):
        names[name].append(y_scores[i])
        vals[name] = int(y_train[i])
    x = []
    y = []
    ns = []
    for name in names:
        x.append(feats_fetch_core(np.array(names[name])))
        y.append(vals[name])
        ns.append(name)
    np.savez('train_test_data/secondtrain_test_%s'%str(version), x=np.array(x), y=np.array(y), name=np.array(ns))


if __name__ == '__main__':
    #second_model_fetch_data(11)
    compute_core('20181105_test_second')

