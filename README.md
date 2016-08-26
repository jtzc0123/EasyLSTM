# LSTM_PittsRoutine
Play with LSTM on Pittsburgh's driving data

---
For inner use of *Ubicomp, HCII, CMU,* with important information omitted.

***Aug/24/2016 Updated:*** Now you are welcome to make use of this lib for your own dataset!

## Preliminaries
Tested on OS X EI Capitan 10.11.5:

+ numpy 1.11.0
+ tensorflow 0.9.0
+ keras 1.0.5
+ scikit-learn 0.17.1
+ googlemaps 2.4.4 (see your model being alive!)

## Pipeline
![](http://epiwork.hcii.cs.cmu.edu/~hanggao/pics/pipeline_lstm_lib.png)

## How to use
### File Structure
It would be much easier if the file structure is specified. So, in this lib, I would like to suggest anyone may concern use the same orgnization form as I do, which is (`*`marked options):
```
${MY_PROJ_NAME}
│   README.md
└───data
│   └───raw
│   └───useful (clean dataset)
│   └───input (save the training and testing piece to evaluate outside the module)
│   └───*prediction (output labels if you want)
└───*logs (For Tensorboard)
└───model (For models)
└───src (The lib)
└───*script (Your customized script to clean up the dataset should be located here)
```

### V1: Easy Used
1. clean up your dataset first before using the lib.
2. `$ cd src`
3. rewrite the `load.py` if necessary for your own need.
4. `$ ipython`
5. `$ from pretraining import divide_save`
6. `$ divide_save(filename, savename, permu_flag, portion, permu_flag, usage_ratio, n_labels, n_seq)` 
7. `quit`
8. now you have the divided (reformatted) dataset in the `data/input` directory
9. `$ python runner.py savename n_seq n_labels n_dimension n_hidden_1 n_hidden_2 epochs tb_on save`
10. have a cup of coffee and wait for the training process.
11. testing thread starts once training finished, all in the `test.py` with ROC_curve, confusion_matrixm, tons of scores and so on.

### V2: Hard Core
1. follow Version 1 instruction to step 7.
2. specify your own topology in `lstm.py`
3. `python runner.py`

## Tips & Suggestions
+ Read LSTM tutorial first at [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
+ Assume you know nothing about LSTMs, then my suggestion is there are few things that you can play with:
    * nb_layers
    * nb_layers_cells
    * nb_seq_length
    * drop_out_rate
+ Check the classic model layout and imitate them as your first topology, you can find them at [Caffe's Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
+ For other details, please go through the code with commentaries.

## Contact
Any questions & suggestions are welcomed. Also, you are encouraged to folk this repo to build your own tool kit.
[Emails](<mailto:cullengao@gmail.com>) and wechats (sc_candi) are the quickest ways to get response.

Salute,

-H
