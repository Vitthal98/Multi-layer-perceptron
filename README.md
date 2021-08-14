# Multi-layer-perceptron
*Implementing a multilayer perceptron classifier from scratch in C*

This project was completed in partial fulfilment of the course MACHINE LEARNING (BITS F464) offered during First Semester 2019-20 at BITS Pilani, Pilani Campus.

The objective of this project was to understand the intricacies of Neural Networks and to get some insights into the working of deep neural networks.

## What is a Multilayer Perceptron?
> A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs using non-linear activation functions.

We implemented a generalized MLP network with any number of layers with different types of activation functions and gradient descent algorithms to compare the accuracy of the
methods over two types of problem.

The types of Activation functions used are **Sigmoid and Tanh**.
The types of gradient descent algorithms used are **regular gradient descent and gradient descent with momentum**.
The learning rate of the network is **0.05** and momentum rate is **0.9**.
The error used is **quadratic error**.

These solutions were implemented in C.

## Binary Classification Problem

### 1. Dataset
   Blood transfusion service center data set is taken for binary classification. The dataset consists of 4 attributes and 1 binary output:
   
   - R (Recency - months since last donation) Numerical Value
   - F (Frequency - total number of donations) Numerical Value
   - M (Monetary - total blood donated in c.c.) Numerical Value
   - T (Time - months since first donation) Numerical Value
   - A binary variable representing whether he/she donated blood in March 2007 (1 stands for donating blood; 0 stands for not donating blood).
    
   The dataset was was cleaned as 2 attributes were totally correlated. Hence only 3 attributes were considered for training. The input values were normalized for better training of the network. 
    
   The training set consisted of 500 values and the test set consists of 248 values. The link of the dataset is https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
    
   The network used a hidden layer of 4 neurons and output layer of 2 neurons.The last is softmaxed for classification.

### 2. Performance
   ![image](https://user-images.githubusercontent.com/51982356/129448156-6b600683-a322-4348-85b3-949a24bf1bcc.png)

## Multiclass Classification Problem

### 1. Dataset
Contraceptive Method Choice is a subset of the 1987 National Indonesia Contraceptive Prevalence Survey. The samples are married women who were either not pregnant or do not
know if they were at the time of interview. The problem is to predict the current contraceptive method choice (no use, long-term methods, or short-term methods) of a woman based on her demographic and socio-economic characteristics.

The dataset consists of 9 attributes and 1 output class as follows:
- Wife's age (numerical)
- Wife's education (categorical)
- Husband's education (categorical)
- Number of children ever born (numerical)
- Wife's religion (categorical)
- Wife's now working? (categorical)
- Husband's occupation (categorical)
- Standard-of-living index (categorical)
- Media exposure (categorical)
- Contraceptive method used (class attribute) 1=No-use, 2=Long-term, 3=Short-term

The dataset used categorical values and converted them to numerical and then normalised the input data for better training. The training set consists of 900 samples and test
set consists of 573 samples.

The link of the dataset is https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice.

The network used a hidden layer of 5 neurons and output layer of 3 neurons. The output was softmaxed for classification.

### 2. Performance
![image](https://user-images.githubusercontent.com/51982356/129448433-32a09b08-ae6a-40a1-a076-b522fee9c39b.png)


### Want to know more?
Please read the [problem statement](https://github.com/Vitthal98/Multi-layer-perceptron/blob/main/Problem%20Statement.pdf) and [project report](https://github.com/Vitthal98/Multi-layer-perceptron/blob/main/ML%20Assignment-1%20Report%202018-19.pdf) to get detailed understanding about the datasets used, methods implemented, results obtained and further discussion.

For any doubts don't hesitate to contact me at vitthalbhandari98@gmail.com

If you find our work helpful, do not forget to :star: the repository!
