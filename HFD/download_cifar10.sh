#!/usr/bin/env bash

# wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz # Linux User
curl "https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz" -o "cifar-10-matlab.tar.gz" # Mac User
tar -xzf cifar-10-matlab.tar.gz
mv cifar-10-batches-mat datasets/
rm cifar-10-matlab.tar.gz
