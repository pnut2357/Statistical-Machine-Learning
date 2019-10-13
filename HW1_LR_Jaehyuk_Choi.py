################################################################################
# FILE: HW1_GNB_Jaehyuk_Choi.py
# AUTHOR: Jaehyuk Choi (1215326372)
# CONTACT INFO: jchoi154@asu.edu
#
# COURSE INFO
# EEE591 Fall 2019
# Homework1_1 09-22-2019
#
# DESCRIPTION
# The program classifies data as image 7 or image 8 from MNIST database.
# Using the Gradient Ascent approach, the weight vector was fianlly update
# in the iterations. Then, the posterior probabilities were directly calculated
# for each training data samples.
################################################################################

import time                   # to measure computational time.
import pandas as pd
import numpy as np
import scipy.io as scio
from scipy import stats
import random as rd
import matplotlib.pyplot as plt

# Constants for data, maximum iteration, and learning rate.
NUM_OF_FEATURES = 2
NUM_OF_TRAIN_SAMPLES_7 = 6265
NUM_OF_TRAIN_SAMPLES_8 = 5851
NUM_OF_TRAIN_SAMPLES = 12116
MAX_ITER = 10000
LEARNING_RATE = 0.0009

# Loads the data from MNIST.
numpy_file = scio.loadmat("mnist_data.mat")

# Separately stores the data.
train_data = numpy_file['trX']
test_data = numpy_file['tsX']
train_labels = np.transpose(numpy_file['trY'])
test_labels = np.transpose(numpy_file['tsY'])

idx_image7 = np.where(train_labels == 0)[0]
#print(idx_image7)
idx_image8 = np.where(train_labels == 1)[0]
#print(idx_image8)

feature1_7 = np.mean(train_data[idx_image7], axis = 1)
feature2_7 = np.std(train_data[idx_image7], axis = 1)

feature1_8 = np.mean(train_data[idx_image8], axis = 1)
feature2_8 = np.std(train_data[idx_image8], axis = 1)

# Reduces the dimension of 784 pixel values into 2 features.
train_data_in_2features = np.zeros((len(train_data),1))

feature1 = np.zeros((len(train_data),1))
feature2 = np.zeros((len(train_data),1))
for index in range(len(train_data)):
    feature1[index] = np.mean(train_data[index])
    feature2[index] = np.std(train_data[index])

# Augmented by adding a dimension of 1s to the original training samples
augmen_dimens = np.ones((len(train_data), 1))

# Concatenates the all data => 12116 x 3 matrix for training data.
train_data_in_2features = np.c_[augmen_dimens, feature1, feature2]

# Same procedure for testing data.
augmen_dimens_test = np.ones((len(test_data), 1))
feature1_test = np.zeros((len(test_data),1))
feature2_test = np.zeros((len(test_data),1))

for index in range(len(test_data)):
    feature1_test[index] = np.mean(test_data[index])
    feature2_test[index] = np.std(test_data[index])

test_data_in_2features = np.c_[augmen_dimens_test, feature1_test, feature2_test ]

# Plots the training data in 2 dimensional space.
plt.plot(feature1_7, feature2_7, 'bo', feature1_8, feature2_8, 'r^')
plt.title('Image Distribution Based on Mean vs. Std')
plt.xlabel('Mean')
plt.ylabel('Standard Deviation')
plt.show()

feature1_7 = feature1_7.reshape(NUM_OF_TRAIN_SAMPLES_7,1)
feature2_7 = feature2_7.reshape(NUM_OF_TRAIN_SAMPLES_7,1)
feature1_8 = feature1_8.reshape(NUM_OF_TRAIN_SAMPLES_8,1)
feature2_8 = feature2_8.reshape(NUM_OF_TRAIN_SAMPLES_8,1)

# Augmentation by adding a dimension of 1s to the original training samples
augmen_dimens_7 = np.ones(NUM_OF_TRAIN_SAMPLES_7).reshape(NUM_OF_TRAIN_SAMPLES_7, 1)
augmen_dimens_8 = np.ones(NUM_OF_TRAIN_SAMPLES_8).reshape(NUM_OF_TRAIN_SAMPLES_8, 1)

vector_x_7 = np.c_[augmen_dimens_7, feature1_7, feature2_7]
vector_x_8 = np.c_[augmen_dimens_8, feature1_8, feature2_8]

# Randomizes the initial weights
w0 = rd.random()
w1 = rd.random()
w2 = rd.random()

# since data X has the range [0, 1], weight should be the same scale.
# thus, the range of random numbers should be between 0 and 1.
# weight is a 3x1 matrix.
weight = np.asmatrix([w0, w1, w2]).T
#print("old weights")
#print(weight)

################################################################################
# Function: sigmoid
# Input:
#    t - an input vector of XW (12116 x 1 vector) (XW = 12116x3 * 3x1 = 12116x1)
# where X = a matrix of feature values (12116 x 3 matrix: 1st_col 2nd_col 3rd_col)
#                                                            1     mean    std
#       W = a vector of weights (3 x 1 vector: 1st_col 2nd_col 3rd_col)
#                                                w_0     w_1     w_2
# Output: returns sigmoid function.
#    1 / [1 + exp(-t)]
#
# NOTE: it requires XW be calculated before calling the function.
################################################################################
def sigmoid (t):
    #print(np.exp(-t))
    return 1. / (1 + np.exp(-t))

################################################################################
# Function: cost_function
# Input:
#    XW - an input vector of XW
#    y - an input vector of training labels (12116 x 1 vector).
#
# Output: returns J.
#    J = 1/m * [ y*log(sigma(XW)) + (1 - y)*log(1-sigma(XW)) ]
#
# NOTE: it requires XW be calculated before calling the function.
################################################################################
def cost_function (XW, y):

    a = np.asarray(y)
    b = np.log(sigmoid(XW))
    b = np.asarray(b)
    temp1 = np.asmatrix(a*b)
    c_ = np.ones((12116,1))
    d_ = np.asmatrix(c_)
    c = np.asarray(d_ - y)
    d = np.log(d_ - sigmoid(XW))
    d = np.asarray(d)
    temp2 = np.asmatrix(c*d)

    J = temp1 + temp2

    return J.sum()/12116 ############

################################################################################
# Function: gradient
# Input:
#    X - a matrix of feature values (12116 x 3 matrix: 1st_col 2nd_col 3rd_col)
#                                                         1     mean    std
#    y - an input vector of training labels (12116 x 1 vector).
#    XW - an input vector of XW (12116 x 1 vector)
# where W = a vector of weights (3 x 1 vector: 1st_col 2nd_col 3rd_col)
#                                                w_0     w_1     w_2
#
# Output: returns gradent.
#    grad = X.T * [sigmoid(XW) - y]
#
# NOTE: it requires XW be calculated before calling the function.
################################################################################
def gradient (X, y, XW):

    temp1 = np.asmatrix(X).T

    a = np.asmatrix(sigmoid(XW))
    b = np.asmatrix(y)
    temp2 = b - a

    grad = np.dot(temp1,temp2)

    return (grad)

################################################################################
# Function: gradient_ascent
# Input:
#    X - a matrix of feature values (12116 x 3 matrix: 1st_col 2nd_col 3rd_col)
#                                                         1     mean    std
#    y - an input vector of training labels (12116 x 1 vector).
#    W - an input vector of weights (3x1 vector [w0; w1; w2])
#    XW - an input vector of XW (12116 x 1 vector)
#    eta - learning rate
#    max_iter - maximum iterations until J converges such that |J_new - J_old| < 0.001
#
# Output: returns a vector of new weights
#    W - a 3x1 vector of new weights [w0; w1; w2]
#
# NOTE: it requires XW be calculated before calling the function.
#       W, J, and grad are updated while J converges (training)
################################################################################
def gradient_ascent (X, y, W, eta, max_iter, XW):
    J = np.zeros((max_iter, 1))
    #print(J.shape)
    converge = False
    current_iter = 0
    while (converge == False):
        W = np.asmatrix(W)
        X = np.asmatrix(X)
        XW = np.dot(X, W)
        XW = np.asmatrix(XW)
        J[current_iter] = cost_function(XW, y) ####
        #h = sigmoid(XW) #
        #error = y - h #
        grad = gradient(X, y, XW) ####
        #W = W + eta*grad ####
        #print("size of error: ", error.shape)

        #print("size of X: ", X.shape)
        W = W + eta*grad

        if (current_iter > 1) and ( np.abs(J[current_iter] - J[current_iter - 1]) < 0.001 or current_iter == max_iter-1 ):
            converge = True
            print("Converged after iterations.")
        #if (current_iter > max_iter-1):
        #    converge = True
        current_iter += 1
        new_weight = W

    return W

XW = np.dot(train_data_in_2features, weight)

start_time = time.time()

new_weight = gradient_ascent(train_data_in_2features, train_labels, weight, LEARNING_RATE, MAX_ITER, XW)

#print("new weights")
#print(new_weight)

probability_list = sigmoid(np.dot(test_data_in_2features, new_weight))

end_time = time.time()

#idx_7 = np.where(probability_list >= 0.5)[0]
#estimated_labels = np.zeros((len(test_data),1))
#print(idx_7)

#estimated_labels[idx_7] = 1

idx_8 = np.where(probability_list < 0.5)[0]

estimated_labels = np.ones((len(probability_list),1))
#print(estimated_labels)
for index in range(len(probability_list)):
    if probability_list[index] < 0.5:
        estimated_labels[index] = 0

#print(estimated_labels)
correct = 0
for index in range(len(probability_list)):
    if estimated_labels[index] == test_labels[index]:
        correct += 1

accuracy = ( (correct) / 2002.0 ) * 100 # correct/number of test data

print("Accuracy from Scratch = %.2f%%" % (accuracy))
print("Computational Learning Time: %s sec" % (end_time - start_time))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

clf = LogisticRegression()
clf.fit(train_data_in_2features, train_labels)
y_pred_sk = clf.predict(test_data_in_2features)
y_pred_sk.reshape(2002, )
confusion_matrix(test_labels, y_pred_sk)
acc_sk = accuracy_score(test_labels, y_pred_sk)
print("Accracy from sklearn = %.2f%" % acc_sk)
