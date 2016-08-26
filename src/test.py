import os
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from keras.models import model_from_json
from sklearn.metrics import roc_curve, auc, confusion_matrix
from math import ceil

model_path = '../models/'
input_path = '../data/input/'

######################################################################
# Read data from divided dataset - `train_` and `test_`
def read_data(savename, n_seq=5, n_labels=12, n_dimension=24, usage_ratio=1):

    train_name = 'train_' + savename + '.csv'
    test_name = 'test_' + savename + '.csv'
    print 'Reading data from: ', train_name, '&', test_name, ', with ratio: ', usage_ratio;
    train_name = input_path + train_name
    test_name = input_path + test_name

    train = np.loadtxt(train_name, delimiter=',')
    test = np.loadtxt(test_name, delimiter=',')

    n_tr = int(ceil(train.shape[0] * usage_ratio))
    n_te = int(ceil(test.shape[0] * usage_ratio))
    train, test = train[:n_tr,:], test[:n_te, :]

    X_train, y_train = train[:, :-n_labels], train[:, -n_labels:]
    X_test, y_test = test[:, :-n_labels], test[:, -n_labels:]

    X_train = X_train.reshape(-1, n_seq, (n_dimension+1)).transpose((0, 2, 1))
    X_test = X_test.reshape(-1, n_seq, (n_dimension+1)).transpose((0, 2, 1))

    print 'Reading Succeed.'
    return X_train, y_train, X_test, y_test

def read_model(savename):

    print '\n[MODEL NAME]: ', savename

    m_path = model_path + 'lstm_architecture_' + savename + '.json'
    model = model_from_json(open(m_path).read())
    w_path = model_path + 'lstm_weights_' + savename + '.h5'
    model.load_weights(w_path)

    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    print 'Reading Success.'
    return model

def test_lstm(model=None, data=None, savename=None):
    print 'Testing LSTM..'

    batch_size = 128

    if data is None:
        X_train, y_train, X_test, y_test = read_data(savename)
    else:
        X_train, y_train, X_test, y_test = data

    print 'Testing with ', X_test.shape[0], ' samples..'

    if model is None:
        for root, dirs, files in os.walk(model_path):
            for modelname in files:
                if 'architecture' in modelname:
                    model = read_model(modelname)
                    acc = model.evaluate(X_test, y_test, batch_size=batch_size)
                    print '\ntesting accuracy: ', acc
    else:
        acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        print '\ntesting accuracy: ', acc

def test_all_metrics(model, data=None, usage_ratio=1):
    if data is None:
        X_train, y_train, X_test, y_test = read_data(usage_ratio=usage_ratio)
    else:
        # You ought to use the same training & testing set from your initial input.
        X_train, y_train, X_test, y_test = data

    y_pred = model.predict_classes(X_test)
    y_ground = np.argmax(y_test, axis=1)
    # y_proba = model.predict_proba(X_test)

    # overall_acc = (y_pred == y_ground).sum() * 1. / y_pred.shape[0]
    precision = sk.metrics.precision_score(y_ground, y_pred)
    recall = sk.metrics.recall_score(y_ground, y_pred)
    f1_score = sk.metrics.f1_score(y_ground, y_pred)
    # confusion_matrix = sk.metrics.confusion_matrix(y_ground, y_pred)
    # fpr, tpr, thresholds = sk.metrics.roc_curve(y_ground, y_pred)

    print "precision_score = ", precision
    print "recall_score = ", recall
    print "f1_score = ", f1_score

    # plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_ground, y_pred)

def plot_roc_curve(y_test, y_proba):
    print 'Ploting roc curve..'
    n_labels = y_proba.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_labels):
        fpr[i], tpr[i], _ = sk.metrics.roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]))
    for i in range(n_labels):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_ground, y_pred, title='Normalized confusion matrix', cmap=plt.cm.Blues):
    print 'Ploting confusion matrix..'
    # Compute confusion matrix
    cm = confusion_matrix(y_ground, y_pred)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    # print(cm_normalized)
    plt.figure()

    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


