# import tensorflow as tf
import numpy as np
import pandas as pd

# You should use your own cleaned-up dataset as default.
#       ***Note: Your dataset should have one row of header.
filename = '../data/useful/pitt_intersection_speed'

######################################################################
# Read data from .csv format file.
# -------------------------------------------------------------------
# input:
#   (str) filename, the filename of data in CSV format
#       ***Note: your first column of csv file should be a indicator as
#       "group_flag" that specify the inner group division. For an instance,
#       your data may not be sampled continuously, which means you would have
#       trouble if you treat them by one giant "sliding window". So, to this
#       point, you may want to use a "group_flag" to tell the library that these
#       are of different group.
# return:
#   (tuple) (X, y),
#       X - (n_sample x n_dimension) array,
#       y - (n_sample x 1) array, raw label information
#
def csv_read(filename=filename):

    print 'Loading raw data from %s..' % filename
    data = pd.read_csv(filename, dtype=str, delimiter=',')
    X = np.asarray(data)[:, 1:-1].astype(float)
    y = np.asarray(data)[:, -1].astype(float).reshape(-1, 1)
    print "[csv_read]: X.shape - ", X.shape, ", y.shape - ", y.shape
    return (X, y)


######################################################################
# Reformat data from .csv format file.
# -------------------------------------------------------------------
# input:
#   (array) X_input, y_input as what it is in your dataset
#   (str) filename, if you don't input your dataset by (X_input, y_input). This func
#       would use csv_read(filename) to fetch the input for you
#   (bool) raw_label_flag, True - use raw label without procession,
#       False - process the raw labels according to your own rule, should be specified
#       by user's input of "n_labels". Or, you wanna do regression.
#   (int) n_seq, the length of LSTM's `window`. You can imagine the model would memorize
#       the pattern exclusively in this sliding window
#   (int) n_labels, the number of your labels, only being used if raw_label_flag=False.
#       In previous work, I process the labels manually in this function. You should rewrite
#       this part if necessary.
#   (bool) permu_flag, indicate permutation or not. For real training process, this group_flag
#       should be on, but you should close this group_flag at the first time or anytime you'd
#       like to check the sanitation of your dataset.
#   (bool) use_y_flag, indicate using previous y as a feature or not.
# return:
#   (tuple) (X, y),
#       X - reformatted X_input, (n_sample-n_seq) x (n_dimension+1) array,
#           ***Note: using (n_dimension+1) for counting previous y value as well,
#               or (n_sample-n_seq) x n_dimension if use_y_flag is False.
#       y - labelized y value, (n_sample-n_seq) x n_labels array.
def csv_reformat(X_input=None, y_input=None, filename=filename, raw_label_flag=False,
    n_seq=5, n_labels=12, permu_flag=True, use_y_flag=True):

    if X_input is None and y_input is None:
        X_, y_ = csv_read(filename)
    else:
        X_ = X_input
        y_ = y_input

    n_samples, n_dimension = X_.shape
    data = np.concatenate((X_, y_), 1)
    group_flag = np.loadtxt(filename, delimiter=',', skiprows=1)[:, 1].astype(int)
    if raw_label_flag is True:
        n_labels = np.unique(y_).shape[0]
    print 'Reformating..'

    _, flag_index = np.unique(group_flag, return_index=True)
    data_seg = []
    y = np.empty((0, n_labels))
    if use_y_flag is True:
        # Add y feature here.
        X = np.empty((0, n_seq * (n_dimension+1)))
    else:
        X = np.empty((0, n_seq * n_dimension))

    # segmenting data according to group_flag
    for i in range(flag_index.shape[0] - 1):
        data_tmp = data[flag_index[i] : flag_index[i + 1]]
        data_seg.append(data_tmp)
    data_seg.append(data[flag_index[flag_index.shape[0] - 1]:])

    for data_tmp in data_seg:
        X_tmp, y_tmp = data_tmp[:, :-1], data_tmp[:, -1].reshape(-1, 1)
        X__ = np.empty((0, X.shape[1]))
        for i in range(X_tmp.shape[0] - n_seq):
            # Add y feature here.
            X_s = data_tmp[i:i+n_seq].reshape(1, -1)
            X__ = np.vstack((X__, X_s))

        if raw_label_flag is True:
            # Use raw label
            y__ = y_tmp[n_seq:]
            y_s = np.zeros((y__shape[0], n_labels))
            for i in range(y__.shape[0]):
                y_s[i, y__[i]] = 1

        else:
            # 12 labels for speed prediction
            # Data related, you may like to add your rule to process your label
            if n_labels == 12:
                y__ = y_tmp[n_seq:].astype(int) / 5
                y_s = np.zeros((y__.shape[0], n_labels), dtype=int)
                for i in range(y__.shape[0]):
                    if y__[i] >= 11:
                        y__[i] = 11
                    y_s[i, y__[i]] = 1

            # 1 label for regresssion
            elif n_labels == 1:
                y__ = y_tmp[n_seq:].astype(float)
                y_s = y__

            # Your own rule to be add here
            # elif n_labels == n:
            #   ...
            #   ...
            #   ...

        y__ = y_s
        # ***Note: vstack & hstack operation is depreciated and 200 times
        # slower than passing value to a numpy.zeros matrix.
        X = np.vstack((X, X__))
        y = np.vstack((y, y__))

    data = np.hstack((X, y))

    if permu_flag:
        np.random.shuffle(data)
    X = data[:, :-n_labels]
    y = data[:, -n_labels:]

    # Normalize here rather than in csv_read().
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    print 'Reformating Succeed.'
    return X, y


#####################################################################
# Testing cases.
if __name__ == '__main__':
    print '[TESTING]: load.py'

    X, y = csv_reformat(filename=filename, raw_label_flag=False,
                        n_seq=5, n_labels=1, permu_flag=False)

    print 'X.shape: ', X.shape
    print 'y.shape: ', y.shape

    print 'X[0] for demo: ', X[0]
    print 'y[0] for demo: ', y[0]

    print 'X[1] for demo: ', X[1]
    print 'y[1] for demo: ', y[1]
