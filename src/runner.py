import sys
import lstm, test, pretraining

######################################################################
# Train the initial data, then test it with multiple evaluators.
# --------------------------------------------------------------------
# input:
#   (str) filename, reading file from the filename. Relative path is appreciated.
#   (str) savename, the filename of data in CSV format. Your divided sets would
#       be saved as "train_$(savename).csv" and "test_$(savename).csv" in "data/input"
#       directory, which is necessary for external testingb before the training.
#   (int) n_seq, the size of your sequence.    
#   (int) n_labels, the number of your labels.
#   (int) n_dimension, the dimension of features, not including y_feature.
#   (list) n_hidden, the topology of your LSTM network.
#   (float) dropout, the dropout rate to control the over-fitting issue.
#       see https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for more informations
#   (int) epochs, training epochs.
#   (bool) tb_on, use Tensorboard to record training curve or not. Save at "logs/".
#       see also, https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html
#   (bool) autosave, auto save the model or not,
#       if False, you would be asked for the save option once training is over.
def train_initial(filename, savename, n_seq, n_labels, n_dimension,
                  n_hidden, dropout, epochs, tb_on, autosave):

    print tb_on, autosave
    usage_ratio = 1

    print '========================= Reformating ========================='
    _, _, _, _ = pretraining.divide_save(filename, savename, n_seq=n_seq, 
      n_labels=n_labels)

    print '========================= Reading ========================='
    X_train, y_train, X_test, y_test = test.read_data(savename=savename,
                                                      n_seq=n_seq,
                                                      n_labels=n_labels,
                                                      n_dimension=n_dimension)
    data = (X_train, y_train, X_test, y_test)

    print '========================= Modeling ========================='
    model = lstm.build_model(n_dimension=n_dimension,
                             n_labels=n_labels,
                             n_seq=n_seq,
                             n_hidden=n_hidden,
                             dropout=dropout)

    print '========================= Training =========================='
    model = lstm.run_network(model=model, data=data, epochs=epochs,
                             usage_ratio=usage_ratio, tb_on=tb_on, autosave=autosave, savename=savename)

    print '========================= Testing =========================='
    test.test_all_metrics(model, data=data, usage_ratio=usage_ratio)


# Pass functional variables from external
if __name__ == '__main__':
    train_initial(
        filename = sys.argv[1],
        savename = sys.argv[2],
        n_seq = int(sys.argv[3]),
        n_labels = int(sys.argv[4]),
        n_dimension = int(sys.argv[5]),
        n_hidden = [int(i) for i in sys.argv[6].split()],
        dropout = float(sys.argv[7]),
        epochs = int(sys.argv[8]),
        tb_on = (sys.argv[9] == 'True'),
        autosave = (sys.argv[10] == 'True')
    )
