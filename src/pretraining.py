import numpy as np
from math import ceil
from sklearn.decomposition import PCA

from load import csv_reformat, csv_read

input_path = '../data/input/'


######################################################################
# Divide the data set into training set and testing set.
# ------------------------------------------------------------
# Depreciated. Use divide_save() instead, only being used when external testing.
# You should save your training and testing set to evaluate the model
# outside the runner module, using model file (.json & .h5) and
# X_train, y_train, X_test, y_test
def divide_data(filename, permu_flag=True, raw_label_flag=False, use_y_flag=True, 
    portion=0.8, usage_ratio=1, n_labels=12, n_seq=5):

    X_, y_ = csv_read(filename)

    X, y = csv_reformat(X_, y_, filename, raw_label_flag, permu_flag=permu_flag, 
        use_y_flag=use_y_flag, n_labels=n_labels, n_seq=n_seq)

    print X.shape, y.shape

    n_samples, n_dimension = X.shape
    n_use_samples = int(ceil(n_samples * usage_ratio))
    train_part = int(ceil(n_use_samples * portion))

    X_train = X[:train_part]
    X_test = X[train_part:n_use_samples]
    y_train = y[:train_part]
    y_test = y[train_part:n_use_samples]

    X_train = X_train.reshape(-1, n_seq, X_train.shape[1]/n_seq).transpose((0, 2, 1))
    X_test = X_test.reshape(-1, n_seq, X_test.shape[1]/n_seq).transpose((0, 2, 1))

    return X_train, y_train, X_test, y_test


######################################################################
# Divide and save training and testing set
# -------------------------------------------------------------------
# input:
#   (str) filename, the filename of data in CSV format
#   (str) savename, the filename of data in CSV format. Your divided sets would
#       be saved as "train_$(savename).csv" and "test_$(savename).csv" in "data/input"
#       directory, which is necessary for external testingb before the training.
#   (bool) raw_label_flag, True - use raw label without procession,
#       False - process the raw labels according to your own rule, should be specified
#       by user's input of "n_labels". Or, you wanna do regression.
#   (bool) permu_flag, indicate permutation or not. For real training process, this flag
#       should be on, but you should close this flag at the first time or anytime you'd
#       like to check the sanitation of your dataset.
#   (bool) pca_flag, indicate PCA or not. My suggestion is never use it unless you would
#       like to visualize dataset.
#   (bool) use_y_flag, indicate using previous y as a feature or not
#   (int) n_components, the number of components if pca_flag is True.
#   (float) portion, the portion of training set from raw dataset.
#   (float) usage_ratio, the ratio of using data. You can play with it at first when you 
#       have a large dataset which is common for deep learning task.
#   (int) n_seq, the length of LSTM's `window`. You can imagine the model would memorize
#       the pattern exclusively in this sliding window
#   (int) n_labels, the number of your labels. In previous work, I process the labels
#       manually in "csv_reformat()" function. You should rewrite this part if necessary.
# return:
#   (tuple) X_train, y_train, X_test, y_test
#       X_train - samples of training set, n_new_train_samples x (n_dimension x n_seq)
#       y_train - labels of training set, n_new_train_samples x n_labels
#       X_test - samples of testing set, n_new_test_samples x (n_dimension x n_seq)
#       y_test - labels of testing set, n_new_test_samples x n_labels
#
#       All above has been copied as "data/input/train_$(savename).csv" and 
#           "data/input/test_$(savename).csv"
#       ***Note: you should use test.read_data to import these 2d array as usable data 
#           for LSTM model later.

def divide_save(filename, savename, raw_label_flag=False, permu_flag=True, pca_flag=False, 
    n_components=8, portion=0.8, usage_ratio=1, n_seq=12, n_labels=5):

    X_, y_ = csv_read(filename)
    if pca_flag:
        pca = PCA(n_components=n_components, whiten=True)
        X_ = pca.fit_transform(X_)

    X, y = csv_reformat(X_, y_, filename, raw_label_flag=raw_label_flag, 
        permu_flag=permu_flag, n_labels=n_labels, n_seq=n_seq)
    print X.shape, y.shape

    n_samples, n_dimension = X.shape
    n_use_samples = int(ceil(n_samples * usage_ratio))
    train_part = int(ceil(n_use_samples * portion))

    X_train = X[:train_part]
    X_test = X[train_part:n_use_samples]
    y_train = y[:train_part]
    y_test = y[train_part:n_use_samples]

    train = np.concatenate((X_train, y_train), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_name = input_path + 'train_' + savename + '.csv'
    test_name = input_path + 'test_' + savename + '.csv'

    print 'Saving divided data..'
    np.savetxt(train_name, train, delimiter=',')
    np.savetxt(test_name, test, delimiter=',')

    return X_train, y_train, X_test, y_test


######################################################################
# Autoencoder, for unsupervised data preprocessing.
# -------------------------------------------------------------------
# Depreciated. 
# You may just skip this step for now.
# def autoencoder(X_train, X_validation, n_hidden_1=256, n_hidden_2=128, batch_size=1024):

    # n_dimension = X_train.shape[1]
    # encoder = tflearn.input_data(shape=[None, n_dimension])
    # encoder = tflearn.fully_connected(encoder, n_hidden_1)
    # encoder = tflearn.fully_connected(encoder, n_hidden_2)

    # decoder = tflearn.fully_connected(encoder, n_hidden_1)
    # decoder = tflearn.fully_connected(decoder, n_dimension)

    # net = tflearn.regression(decoder, optimizer='adam',
                            # learning_rate=0.001,
                            # loss='mean_square',
                            # metric=None)

    # model = tflearn.DNN(net, tensorboard_verbose=0)
    # model.fit(X_train, X_train, n_epoch=32, validation_set=(X_validation, X_validation),
            # run_id='mean_square', batch_size=batch_size)

    # return encoder


######################################################################
# Testing cases.
# if __name__ == '__main__':

#     filename = 'YOUR_DATA_SET_NAME'
#     # Labels are not necessary here, for we only reconstruct raw feature space
#     X_train, _, X_test, _ = divide_data(filename, permu_flag=True)

#     encoder = autoencoder(X_train, X_test)
#     encoding_model = tflearn.DNN(encoder)

#     print('\nTest encoding of X[0]:')
#     print(encoding_model.predict([X_test[0]]))

