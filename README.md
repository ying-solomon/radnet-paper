# RadNet
Deep neural network for climate science radiation functions

## Train
To train a model use train.py
Example:

```
python ./train.py --data_dir ./samples/ --test_dir ./samples/
```

In this case, we pass the data using the option "--data_dir" and test with "--test_dir". inside data are many folders each one containing samples of the dataset, so no matter how many nested dirs, the reader will reach the .nc files.
In this repository, we have included one test sample of the nc file, i.e., radiation_1929_m01_c44_107_v0.nc
The model will generate a dir in which it places the models and the summaries of each model. The model will store only the latest models.


## TensorBoard
Just pass the directory that contains the models and log summaries.

```
tensorboard --logdir ./logdir/
```

## Project features
The model is able to save the model, load a model and keep training it even with other hyperparameters, log loss summaries tensorboard and tune other features. For more help: 

```
python train.py --help
```

## File loader
The fileReader.py includes various methods of loading the ERA-interium dataset, with/out interpolation/extrapolation
of a static pressure grid. And different ways of preparing ERA-interium dataset to be the input of the neural networks.
Make sure that the trained model takes in the same input structure when inferencing.

## The Model
The model is defined in radiation/model.py. Since in the paper, we have a lot of configurations of the models with varying number of convolutional layers and the input size.
Thus, it is up to the user to modify the model structure to suits his/her needs by just commenting/uncomment the code or modifying a couple lines of code. 
We have prepared one of the model architecture used in the paper.

## Tuning the Architecture
It is up to the user to tune the architecture of RadNet to use various NN model structures and inputs. In order to do this, the user needs to adapt the fileReader.py to do different data preprocessings and feed in different dimensions of the data defined in model.py.
Also, the radnet.py model needs to take the correct output dimensions from the model output. In this way, we can reproduce the experiments in the paper, i.e., various NN models and different data preprocessing methods.

## Training and Inferencing
The file train.py generates when finished, or interrupted with ctrl+c generates a file called "graph_frozen_radnet.pb" (in logdir/train/<date>/graph_frozen_radnet.pb) that contains the final state of the model that can be loaded for inferencing the model. 

The file test_radnet_script.ipynb contains an example on how to use the inferencing library that builds up the Tensorflow graph into memory and can be fetched in further calls.
It also has various helper functions to calculate statistics of the input dataset and predictions.
A small NN model, i.e., 5 fully connected layer, trained on ERA-interium dataset is included for testing.

## single column model
The file test_equilibrium.ipynb compares radnet radiation prediction with climt rrtmg radiation calculation. They
can also drive the single column model to equilibrium. A small NN model used to run the single column simulation is
included in the release. The code should run out of the box.