# EasyLSTM
An ***LSTM Library*** build on Keras and Tensorflow for Adepter Use of Sequential Sensor Data

## Goal
>Why is it people think deep learning as something grandiose and being frightened to try something **simple and beautiful**?

## Preliminaries
Tested on OS X EI Capitan 10.11.5:

+ numpy 1.11.0
+ tensorflow 0.9.0
+ keras 1.0.5
+ scikit-learn 0.17.1

## Pipeline
![](http://epiwork.hcii.cs.cmu.edu/~hanggao/pics/pipeline_lstm_lib.png)

## How to use
### File Structure
It would be much easier if the file structure is specified. So, in this lib, I would like to suggest anyone may concern use the same organization form as I do, which is (`*`marked options):
```
EasyLSTM
├── LICENSE.md
├── README.md
├── data
│   ├── input
│   ├── prediction
│   ├── raw
│   └── useful
│       └── pitt_intersection_speed.csv
├── logs
├── models
├── script
└── src
    ├── load.py
    ├── lstm.py
    ├── pretraining.py
    ├── runner.py
    └── test.py
```

### Getting Started
1. clean up your dataset first before using the lib.
2. put it into `data/useful` directory.
2. `$ cd src/`
3. rewrite the `load.py` if necessary for your own need.
9. `$ python runner.py filename savename n_seq n_labels n_dimension n_hidden dropout epochs tb_on autosave`
10. have a cup of coffee and wait for the training process.
11. testing thread starts once training finished, all in the `test.py` with ROC_curve, confusion_matrix, tons of scores and so on.

### Quick Example
In `data/useful`, a dummy dataset has been placed for you to do a quick example. It's a small fraction of my previous work about understanding driver's behaviour in Pittsburgh's intersections.

Using `runner.py`, you can simply start your first demo using *EasyLSTM*. You can call externally from terminal, passing args to `sys` for calling `train_initial()` function.
```python
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
                  n_hidden, dropout, epochs, tb_on, autosave)
```

```bash
$ cd src/
$ python runner.py '../data/useful/pitt_intersection_speed.csv' 'speed_5_12_24' 5 12 24 '64 16 16' 0.3 100 False False
```

### Where to Customize?
You are encouraged to modify this lib as what you want it to be.

For now, there are two parts of library that has been left out for simple extension:

+ *load*.**csv_reformat()**
> specify your rules of making labels here.
+ *lstm*.**build_model()**
> change your LSTM's detailed information if what existed cannot satisfy your acdemical demand.

## Tips & Suggestions
+ Read LSTM tutorial first at [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
+ Assuming you know nothing about LSTMs, then my suggestion is there are few things that you can play with:
    * nb_layers
    * nb_layers_cells
    * nb_seq_length
    * drop_out_rate
+ Check the classic models' layout and imitate them as your first topology, you can find them at [Caffe's Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
+ Use power of two as the number of every layers' cell, which can speed up training process, a great deal.
+ Check TensorBoard at `logs/` directory every time you finish a training. For more detailed usage, see also <https://www.tensorflow.org/versions/r0.10/how_tos/graph_viz/index.html>.

## Documentations 
>Please check for commentaries if you want to look for further details.

These few functions may be useful if you know the meaning of args: 

+ *load*.**csv_reformat()**
```python
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
    n_seq=5, n_labels=12, permu_flag=True, use_y_flag=True)
```
+ *pretraining*.**divide_save()**
```python
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
    n_components=8, portion=0.8, usage_ratio=1, n_seq=12, n_labels=5)
```
+ *lstm*.**build_model()**
```python
######################################################################
# Build model topology and compile.
# --------------------------------------------------------------------
# input:
#   (bool) use_y_flag, indicate using previous y as a feature or not.
#   (int) n_dimension, the dimension of features, not including y_feature.
#   (int) n_labels, the number of your labels.
#   (int) n_seq, the size of your sequence,
#   (list) n_hidden, the list of your LSTM topology, should be tuned for the best.
#   (float) dropout, the dropout rate to control the over-fitting issue.
#       see https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf for more informations
# return:
#   (Keras.model) model, compiled model which is ready for training.

def build_model(use_y_flag=True, n_dimension=24, n_labels=12, n_seq=5, 
    n_hidden=[256, 64], dropout=0.3)
```
+ *lstm*.**build_model()**
```python
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
                usage_ratio=1, tb_on=False, autosave=False)
```

## Credit
- 08/26/2016 **Ubicomp Lab**, HCII, CS, CMU.


## Contact
Any questions & suggestions are welcomed. Also, you are encouraged to folk this repo to build your own tool kit.
[Emails](<mailto:cullengao@gmail.com>) and Wechats (sc_candi) are the quickest ways to get response.

Salute,

-H
