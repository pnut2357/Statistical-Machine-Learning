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
# Using the multivariate normal distribution equation, posterior probabilities
# for each training data samples were calculated with prior and conditional
# probabilities.
################################################################################

import time
import pandas as pd
import numpy as np
import scipy.io as scio
from scipy import stats
import statistics as stat
# Number of dimension
NUM_OF_FEATURES = 2

# Loads the data from MNIST.
numpy_file = scio.loadmat("mnist_data.mat")

# Separately stores the data.
train_data = numpy_file['trX']
test_data = numpy_file['tsX']
train_labels = np.transpose(numpy_file['trY'])
test_labels = np.transpose(numpy_file['tsY'])
# Checks to see the data with labels
# (i.e. checks how many data are for image 7 and image 8)
train_data_conca = pd.DataFrame.from_records(np.concatenate((train_labels, train_data), axis = 1))
test_data_conca = pd.DataFrame.from_records(np.concatenate((test_labels, test_data), axis = 1))
T_dic = dict(train_data_conca.iloc[:,0].value_counts())

T_dic2 = dict(test_data_conca.iloc[:,0].value_counts())

# Checks the indices for image 7 and image 8 to calculate
# mean and standard deviation.
idx_image7 = np.where(train_labels == 0)[0]
#print(idx_image7)
idx_image8 = np.where(train_labels == 1)[0]
#print(idx_image8)

feature1_7 = np.mean(train_data[idx_image7], axis = 1)
feature2_7 = np.std(train_data[idx_image7], axis = 1)

feature1_8 = np.mean(train_data[idx_image8], axis = 1)
feature2_8 = np.std(train_data[idx_image8], axis = 1)

# Assigns the mean and stadard deviation for images 7 and 8
# to new vectors separately.
train_mean_vector_image7 = np.zeros([len(feature1_7)], float)
train_std_vector_image7 = np.zeros([len(feature1_7)], float)

for index in range(len(feature1_7)):
    train_mean_vector_image7[index] = np.mean(feature1_7[index])
    train_std_vector_image7[index] = np.std(feature1_7[index])

train_mean_vector_image8 = np.zeros([len(feature1_8)], float)
train_std_vector_image8 = np.zeros([len(feature1_8)], float)

for index in range(len(feature1_8)):

    train_mean_vector_image8[index] = np.mean(feature1_8[index])
    train_std_vector_image8[index] = np.std(feature1_8[index])

# Concatenates two vectors of feature 1 and feature 2.
train_data_in_2features_image7 = np.c_[train_mean_vector_image7, train_std_vector_image7]

train_data_in_2features_image8 = np.c_[train_mean_vector_image8, train_std_vector_image8]

train_data_in_2features = np.vstack((train_data_in_2features_image7, train_data_in_2features_image8))

# Calculates the mean and stadard deviation for each feature
mean_feature1_image7 = np.mean(feature1_7)
std_feature1_image7 = np.std(feature1_7)
mean_feature2_image7 = np.mean(feature2_7)
std_feature2_image7 = np.std(feature2_7)

mean_for_likelihood_7 = np.matrix([[mean_feature1_image7], [mean_feature2_image7]])

mean_feature1_image8 = np.mean(feature1_8)
std_feature1_image8 = np.std(feature1_8)
mean_feature2_image8 = np.mean(feature2_8)
std_feature2_image8 = np.std(feature2_8)

# mu for likelihood for image 8
mean_for_likelihood_8 = np.matrix([[mean_feature1_image8], [mean_feature2_image8]])


# Reduces the dimension for testing datasets and cocatenates
# them into a 2002 x 2 matrix
test_mean_vector = np.zeros([len(test_data)], float)
test_std_vector = np.zeros([len(test_data)], float)

for index in range(len(test_data)):
    test_mean_vector[index] = np.mean(test_data[index])
    test_std_vector[index] = np.std(test_data[index])

test_data_in_2features = np.c_[test_mean_vector, test_std_vector]

# Calculates the covariance matrix and its determinants.

covar_image7 = np.matrix([[std_feature1_image7**2, 0], [0, std_feature2_image7**2]])# covariance(NUM_OF_FEATURES, std_vector_image7)
covar_image8 = np.matrix([[std_feature1_image8**2, 0], [0, std_feature2_image8**2]]) # covariance(NUM_OF_FEATURES, std_vector_image8)

det_of_covariance_image7 = np.linalg.det(covar_image7)
det_of_covariance_image8 = np.linalg.det(covar_image8)

################################################################################
# Function: multivariate_norm_dist
# Input:
#    num_of_dimen - number of features: 2
#    cov - covariance matrix (2 x 2 matrix because we have 2 features).
#    x - a matrix of feature values (12116 x 2 matrix: 1st_col 2nd_col)
#                                                       mean     std
#    mu - an input vector of mean (12116 x 1 vector)
#
# Output: returns multivatiate normal distribution
#    result - 1 / [(2 pi)^(d/2)*cov^(1/2)] *exp(-1/2(x-mu).T * inv(cov) * (x-mu))
#
################################################################################
def multivariate_norm_dist(num_of_dimen, cov, x, mu):
    alpha = 1 / ( (2 * np.pi)**(num_of_dimen / 2) * np.sqrt(np.linalg.det(cov)) )
    x_i = np.matrix(x.reshape(2,1))
    mu = np.matrix(mu)
    result = alpha * ( np.exp( -1/2 * np.transpose(x_i - mu) * np.linalg.inv(cov) * (x_i - mu) ) )
    return result

#print(mean_for_likelihood_7)
#l = np.matrix(test_data_in_2features[0].reshape(2,1))
#print(multivariate_norm_dist(NUM_OF_FEATURES, covar_image7, test_data_in_2features[100], mean_for_likelihood_7))

prob_Y_givenX_7 = np.zeros([len(test_data)], float) #.reshape(len(test_data_in_2features), 1)
prob_Y_givenX_8 = np.zeros([len(test_data)], float)
predicted_label = np.zeros([len(test_data)], float)


prior_prob_7 = 6265 / 12116
prior_prob_8 = 1 - prior_prob_7
start_time = time.time()
for index in range(len(test_data)): #len(test_data)):
    prob_Y_givenX_7[index] = prior_prob_7 * multivariate_norm_dist(NUM_OF_FEATURES, covar_image7, test_data_in_2features[index], mean_for_likelihood_7)
    prob_Y_givenX_8[index] = prior_prob_8 * multivariate_norm_dist(NUM_OF_FEATURES, covar_image8, test_data_in_2features[index], mean_for_likelihood_8)
    if prob_Y_givenX_7[index] < prob_Y_givenX_8[index]:
        predicted_label[index] = 1

end_time = time.time()
print(prior_prob_7)
print(prior_prob_8)

pred_labels = predicted_label.reshape(2002,1)


# Calculates the accuracy
correct = 0
print()
for index in range(len(test_data)):
    if pred_labels[index] == test_labels[index]:
        correct += 1

accuracy = ((correct) / 2002.0*100) # correct/number of test data

print("Accuracy from Scratch = %.2f%%" % (accuracy))
print("Computational Learning Time: %s sec" % (end_time - start_time))

print("train data: ", train_data_in_2features.shape)
print("train labels: ", train_labels.shape)
print("test data: ", test_data_in_2features.shape)
print("test labels: ", test_labels.shape)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

clf = GaussianNB()
clf.fit(train_data_in_2features, train_labels)
y_pred_gnb = clf.predict(test_data_in_2features)
#y_pred_gnb.reshape(len(test_labels),)
confusion_matrix(test_labels, y_pred_gnb)
acc_gnb = accuracy_score(test_labels, y_pred_gnb)
print("Accracy from sklearn =", (acc_gnb))
