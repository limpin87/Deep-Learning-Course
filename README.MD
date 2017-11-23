# Deep-Learning-Course

This repository contains all of the projects which have been submitted in part for the Deep Learning Developer Course organized by RedDragon AI.

Below are brief descriptions of each project carried out.

## 1. [Dense NN classification Problem](https://github.com/limpin87/Deep-Learning-Course/tree/master/DenseNN_Project).

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
4. Using the binary representation, the network similarly is able to achieve accuracies in the range of ~90%. However, the various optimal points achieved during the training period is unstable resulting in fluctuations in performance as the number of epoches.

## 2. [Convolutional Neural Network (CNN) Problem](https://github.com/limpin87/Deep-Learning-Course/tree/master/CNN_Project)

### Problem Statement
In this project, CNNs are used for image classification. The QuickDraw dataset supplied by Google is used in this use-case.
In total there are 345 classes of Doodle images avaliable. Due to computational limits, a total of 100 cllsses will be used with 8000 images as the training set and 2000 images as the test set.

### Solutions:
In total two applications (one possible future extention) were experimented using this dataset.
1. A simple [classification network](https://github.com/limpin87/Deep-Learning-Course/blob/master/CNN_Project/CNN_google_doodle_Batch_norm.ipynb) is created. The input is the normalized 28*28 image vectorized into a 784 long vector. The output will be a 100 class vector. 
2. A [image search algorithm](https://github.com/limpin87/Deep-Learning-Course/blob/master/CNN_Project/Image_Retrieval_for_QuickDraw_CNN.ipynb) was created using the same network developed in the earlier notebook.
3. Future Extension: Make use of a more complex network (e.g. VGG19) for image search. This would require upsampling the images to have the pre-defined input dimensions.

### Results/Observations
1. 
