# Deep-Learning-Course

This repository contains all of the projects which have been submitted in part for the Deep Learning Developer Course organized by RedDragon AI.

Below are brief descriptions of each project carried out.

## 1. Dense NN classification Problem [Link](https://github.com/limpin87/Deep-Learning-Course/tree/master/DenseNN_Project).

### Problem Statement
In this project, a dense Neural Network was created to solve a regression problem as a classifcation problem.
- The network is supposed replicate the action of summing two integer values between (0-100) and return the result as one of the possible 201 classes.
- To test the generalizability of the network, all training datapoints which summed to the value of 50 are removed.

### Solutions
In total 2 notebooks were created each testing a different solution.
1. The [first notebook](https://github.com/limpin87/Deep-Learning-Course/blob/master/DenseNN_Project/Project%201.ipynb) adpots a straight forward approach to tackling the problem. The outputs are one-hot encoded to represent a classification problem. 
2. The [second notebook](https://github.com/limpin87/Deep-Learning-Course/blob/master/DenseNN_Project/Project%201_binary_rep.ipynb) takes a different approach. Both the inputs and outputs are encoded using a binary representaton. This reduces the dimension of the problem significantly.

### Results / Observations
1. In the first approach, it was observed that a significant number of training datapoints is need to achieve a good prediction accuracy. 
2. A significant number of training epoches is also requried to obtain a good training and validation accuracy.
3. In general the network is able to correctly predict some of the unseen dataset (i.e. output of 50) but overall the testing accuracy is lower than the benchmark accuracy expected.
4. Using the binary representation, the 