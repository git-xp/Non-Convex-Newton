# Second-Order Optimization for Non-Convex Machine Learning

This repository contains Matlab code that produces all the experimental results in the paper: [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827).

Specifically, multilayer perceptron(MLP) networks and non-linear least squares(NLS) are the two non-convex problems considered.

## Usage

### MLP networks
- <code>[HFD](./HFD)</code>: This folder contains all the source code for implementing the MLP problems that are considered in the paper. 
- <code>[HFD/algorithms](./HFD/algorithms)</code> contains the implementation of (sub-sampled) trust-region, gaussian-newton, momentum sgd algorithms.
- <code>[HFD/mdoel](./HFD/model)</code> constains the implementation of general neural network framework.
- <code>[HFD/cifar_classification](./HFD/cifar_classification)</code>: a demo function that uses various algorithms to train a 1-hidden-layer network on cifar10.
- <code>[HFD/mnist_autoencoder](./HFD/mnist_autoencoder)</code>: a demo function that uses various algorithms to train autoencoder on mnist. 

#### Example 1: Cifar10 Classification
```
Download Cifar10 dataset from here: https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
```
or
run the command
```
bash download_cifar10.sh
```
In the Matlab Command Window, run
```
# check details of the function for different configurations
>> result = cifar_classifcation
```

#### Example 2: mnist Autoencoder
In the Matlab Command Window, run
```
# check details of the function for different configurations
>> result = mnist_autoencoder
```

### NLS
- <code>[nls](./nls)</code>: This folder contains all the source code for implementing the binary linear classification task using square loss (which gives a non-linear square problem). 
- <code>[nls/algorithms](./nls/algorithms)</code> contains the implementation of (sub-sampled) TR, ARC, GN, GD, LBFGS algorithms for non-linear least squares.

#### Example 3: NLS on ijcnn1
```
Download 'ijcnn1' dataset from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
```
or run the command
```
bash download_ijcnn1.sh
```
In the Matlab Command Window, run
```
# this will generate the plots of all algorithms.
# check the details of the function for more options.
>> blc_demo('ijcnn1')
```


## References
- Peng Xu, Farbod Roosta-Khorasani and Michael W. Mahoney, [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827), 2017
- Peng Xu, Farbod Roosta-Khorasani and Michael W. Mahoney, [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164), 2017

