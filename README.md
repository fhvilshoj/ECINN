### Running the code
# ECINN: Efficient Counterfactuals from Invertible Neural Networks

This is the official code base for the paper [ECINN: Efficient Counterfactuals
from Invertible Neural Networks](https://arxiv.org/abs/2103.13701).

## Practical stuff

I have made all entrypoints accessible through the main.py file to remove clutter.

```bash
$> python main.py -h
``` 

Will show you the way.
There are three modules. 
1. First, the `train` module should be used to train an IB-INN model.
2. Next, the `list` module can be used to display indexed output folders, for
   ease of indexing when choosing model to explain.
3. Finally, the `counterfactual` module computes counterfactual examples and
   stores them in the directory specified in `config.ini`.


### Installation and submodule
The python requirements are listed in `requirements.yml` and can be installed with conda as follows:
```bash
$> conda env create -f requirements.yml
```
This will produce conda environment `ecinn` with necessary dependencies.
It will also make a directory in the root named `src`. This contains the
[FrEIA](https://github.com/VLL-HD/FrEIA) framework for invertible neural
networks in PyTorch.

Furthermore, this code makes use of the
[IB-INN](https://github.com/VLL-HD/IB-INN) code base.  To clone the code into
the submodule directory, run the following command.

```bash
$> git submodule update --init
```

### The `config.ini` file

When counterfactuals are computed, they are stored as separate files 
that are located in subdirectories with the root being specified in the 
`config.ini` file.

## FakeMNIST
In the paper, we introduce a new Dataset called the FakeMNIST dataset, where 
we took the [MNIST](http://yann.lecun.com/exdb/mnist/) data, scrambled the
labels and added a little dot in the top-left corner to indicate the new label.

`dataset/fakemnist.py` contains a pytorch Dataloader, which deterministically 
scrambles labels and draws the dots in the top-left corner.





