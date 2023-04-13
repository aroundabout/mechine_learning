# mechine learning homework 1

>
    name: 
    number : 
>

## Preprocess

1. conda create -n python=3.9.16
2. conda activate tf
3. conda install --yes --file requirements.txt

The file structure should keep the same. The checkpoints extension name is .h5
>
mechine_learning/
├── checkpoints
│   └── mnist_model_weights.h5
├── conda_requirements.txt
├── config-default.yaml
├── init.sh
├── LICENSE
├── model
│   ├── __init__.py
│   └── Mnist_model.py
├── README.md
├── requirements.txt
├── runner
│   ├── __init__.py
│   └── train.py
├── test.sh
├── train.sh
└── utils
    ├── const.py
    ├── __init__.py
    └── tools.py
>

## How to run

1. how to train 
   1. cd mechine_learning
   2. sh train.sh
2. how to test
   1. cd mechine_learning
   2. sh test.sh

## result 

1. Test loss: 0.02574940212070942
2. Test accuracy: 0.9905999898910522
