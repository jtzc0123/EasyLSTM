import time
import os

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import TensorBoard
# from keras.regularizers import l2

from test import read_data


######################################################################
# Build model topology and compile.
# --------------------------------------------------------------------
# input:
#   (bool) use_y_flag, indicate using previous y as a feature or not.
#   (int) n_dimension, the dimension of features, not including y_feature.
#   (int) n_labels, the number of your labels.
#   (int) n_seq, the size of your sequence.
#   (list) n_hidden, the list of your LSTM topology, should be tuned for the best.
#   (float) dropout, the dropout rate to control the over-fitting issue.
#       see https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for more informations
# return:
#   (Keras.model) model, compiled model which is ready for training.
def build_model(use_y_flag=True, n_dimension=24, n_labels=12, n_seq=5, 
    n_hidden=[256, 64], dropout=0.3):

    print 'Building LSTM model..'
    model = Sequential() 
    if use_y_flag is True:
        n_dimension += 1
    n_labels = n_labels
    n_seq = n_seq
    # layers = [n_seq, n_hidden_1, n_hidden_2, n_labels]
    layers = [n_seq] + n_hidden + [n_labels]
    print layers

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        input_length=n_dimension,
        return_sequences=True,
        # W_regularizer=l2(l=0.01),
        # b_regularizer=l2(l=0.01)
    ))

    for i in range(2, len(layers)-2):
        model.add(LSTM(
            layers[i],
            input_length=n_dimension,
            return_sequences=True,
            # W_regularizer=l2(l=0.01),
            # b_regularizer=l2(l=0.01)
        ))

    model.add(LSTM(
        layers[-2],
        input_length=n_dimension,
    ))

    model.add(Dense(
        output_dim=layers[-2],
        activation='relu',
    ))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[-2],
        activation='relu',
    ))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[-1],
    ))

    if n_labels != 1:
        model.add(Activation("softmax"))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        print 'Compiling..'
    else:
        model.add(Activation("linear"))

        model.compile(
            loss='mse',
            optimizer='rmsprop'
        )

    print 'Compiling Succeed.'
    return model


######################################################################
# Start the training process.
# --------------------------------------------------------------------
# input:
#   (str) savename, after this func, your model would be save at "models/lstm_architecture_$(savename).json"
#       and "models/lstm_weights_$(savename).h5"
#   (Keras.model) model, input the compiled model. You can input whatever model
#       you want once it has been compiled.
#   (tuple) data, (X_train, y_train, X_test, y_test) which should be import as same format
#       as in pretraining.divide_save()
#   (int) epochs, training epochs.
#   (float) usage_ratio, the ratio of using data. You can play with it at first when you 
#       have a large dataset which is common for deep learning task.
#   (bool) tb_on, use Tensorboard to record training curve or not. Save at "logs/".
#       see also, https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html
#   (bool) autosave, auto save the model or not,
#       if False, you would be asked for the save option once training is over.
# return:
#   (Keras.model) model, model that can be used for prediction process.

def run_network(savename, model=None, data=None, epochs=100, \
                usage_ratio=1, tb_on=False, autosave=False):
    epochs = epochs
    batch_size = 128
    start_time = time.time()
    print 'Running LSTM, start time: ', start_time

    if data is None:
        # Add your dataset infos here if you don't want to use `runner` to
        # train you LSTM model.
        dummyname = 'YOUR_OWN_DATASET'
        X_train, y_train, X_test, y_test = read_data(train_name='train_' + dummyname,
                                                     test_name='test_' + dummyname,
                                                     usage_ratio=usage_ratio)
    else:
        X_train, y_train, X_test, y_test = data

    print 'Modeling with ', X_train.shape[0], ' training samples & ', \
        X_test.shape[0], ' testing samples..'

    if model is None:
        model = build_model()

    # EarlyStopping feature is available,
    # However, my suggestion is to tune out the optimal parameter set, then use this
    # callback to accelerate your training process.
    # earlyStopping=EarlyStopping(monitor='val_loss',
                                # patience=5,
                                # verbose=1,
                                # mode='auto')

    callbacks = []
    if tb_on is True:
        callbacks.append(TensorBoard(log_dir='../logs'))
        print "Training Mode: TENSORBOARD ON.."
    else:
        print "Training Mode: TENSORBOARD OFF.."
    try:
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            nb_epoch=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print 'Training over, total executing time: ', time.time() - start_time

    print 'Training over, total executing time: ', time.time() - start_time

    if autosave or raw_input('Save the model? (y/n): ') == 'y':
        # Save the model into `.json` & `.h5` files,
        # in which JSON file contains the layout of your model,
        # and H5 file contains the weights of your model
        # ----------------------------------------
        # Note that: `.json` file will be overwriten without warning.
        json_string = model.to_json()

        filename1 = 'lstm_architecture'
        filename2 = 'lstm_weights'

        cnt = 1
        model_path = '../models/'
        for root, dirs, files in os.walk(model_path):
            for filename in files:
                if 'lstm_architecture' in filename:
                    cnt += 1
        filename1 = model_path + filename1 + '_' + savename + '.json'
        filename2 = model_path + filename2 + '_' + savename + '.h5'

        print 'Saving LSTM model into .json & .h5 at: %s & %s' \
            % (filename1, filename2)
        open(filename1, 'w').write(json_string)
        model.save_weights(filename2)

    # Raw testing
    acc_train = model.evaluate(X_train, y_train, batch_size=128)
    acc_test = model.evaluate(X_test, y_test, batch_size=128)
    print '\ntraining accuracy: ', acc_train
    print 'testing accuracy: ', acc_test

    print '\nTraining down.'
    return model
