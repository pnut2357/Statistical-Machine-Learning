###############################################################################
# FILE: project1_B.py
# AUTHOR: Jaehyuk Choi (1215326372)
# CONTACT INFO: jchoi154@asu.edu
#
# COURSE INFO
# EEE591 Fall 2019
# Homework5 10-10-2019 / Time: 9:40am - 10:30am MWF
#
# DESCRIPTION
# THe program reads the input data of heart1.csv and outputs the best accuracies
# for each learning algorithm to compare and determine which model can be the best
# one for heart disease prediction. For resulting the best accuracies, parameters
# are tuned manually with plotting accuracy as a function of several parameters.
# Since the given data is not big enough, it was divided into 70% of training
# dataset and 30% of testing dataset and accuracy was recalculated after combining
# training and testing data.
###############################################################################

import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn import datasets                           # read the data sets
import pandas as pd                                    # import and analyze data
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
# algorithms - PPN, LR, SVM, DT, RF, KN =========================
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# ===============================================================
from sklearn.metrics import accuracy_score             # grade the results


heart = pd.read_csv('heart1.csv')           # Loads the data set
X = heart.values[:,0:13]                    # Separates 13 features from the database
y = heart.values[:,13]                      # Separates the output from the database

# Splits the raw data into 70% training dataset and 30% testing dataset
# random_state allows the split to be reproduced
# stratify=y not used in this case
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# scale X by removing the mean and setting the variance to 1 on all features.
# the formula is z=(x-u)/s where u is the mean and s is the standard deviation.
# (mean and standard deviation may be overridden with options...)

sc = StandardScaler()                      # create the standard scalar
sc.fit(X_train)                            # compute the required transformation
X_train_std = sc.transform(X_train)        # apply to the training data
X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!

print('Number in test ',len(y_test))

################################################################################
# Perceptron (PPN)
# ===== Best Parameter Tuned =====
# eta0 (learning rate): 0.001
# max_iter: 1000
# alpha: 5
# number of iteration with no change: 5
# toleance: 10^(-3)
################################################################################
ppn = Perceptron(alpha=5, max_iter=1000, n_iter_no_change=5, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=False)
ppn.fit(X_train_std, y_train)              # do the training

print()

print("============================= Perceptron =============================")
y_pred_ppn = ppn.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_ppn).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_ppn))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_ppn = np.vstack((X_train_std, X_test_std))
y_combined_ppn = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_ppn))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_ppn = ppn.predict(X_combined_std_ppn)
print('Misclassified combined samples: %d' % (y_combined_ppn != y_combined_pred_ppn).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_ppn, y_combined_pred_ppn))

print()

################################################################################
# Logistic Regression (LR)
# ===== Best Parameter Tuned =====
# C as the inverse of the regularization strength: 2.0
# solver: liblinear
# multi class: ovr
# max iteration: 10
################################################################################
lr = LogisticRegression(C=2.0, solver='liblinear', multi_class='ovr', random_state=0, max_iter=10)
lr.fit(X_train_std, y_train)                # apply the algorithm to training data
print("========================= LogisticRegression =========================")
y_pred_LR = lr.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_LR).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_LR))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_LR = np.vstack((X_train_std, X_test_std))
y_combined_LR = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_LR))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_LR = lr.predict(X_combined_std_LR)
print('Misclassified combined samples: %d' % (y_combined_LR != y_combined_pred_LR).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_LR, y_combined_pred_LR))

print()

################################################################################
# Support Vector Machine (SVM)
# ===== Best Parameter Tuned =====
# kernal - linear
# C as the penalty parameter: 2.0
# gamma: 0.03
################################################################################
svm = SVC(kernel='linear', gamma=0.03, C=2.0, random_state=0, max_iter = -1)
svm.fit(X_train_std, y_train)                      # do the training
print("================== Support Vector Machine (linear) ==================")
y_pred_SVM_lr = svm.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_SVM_lr).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_SVM_lr))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_SVM_lr = np.vstack((X_train_std, X_test_std))
y_combined_SVM_lr = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_SVM_lr))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_SVM_lr = svm.predict(X_combined_std_SVM_lr)
print('Misclassified combined samples: %d' % (y_combined_SVM_lr != y_combined_pred_SVM_lr).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_SVM_lr, y_combined_pred_SVM_lr))

print()

################################################################################
# Support Vector Machine
# ===== Best Parameter Tuned =====
# kernal - rbf
# C as the penalty parameter: 2.0
# gamma: 0.09
################################################################################
svm_rbf = SVC(kernel='rbf', gamma=0.09, C=2.0, random_state=0, max_iter = -1)
svm_rbf.fit(X_train_std, y_train)                      # do the training
print("==================== Support Vector Machine (rbf) ====================")
y_pred_SVM_rbf = svm_rbf.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_SVM_rbf).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_SVM_rbf))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_SVM_rbf = np.vstack((X_train_std, X_test_std))
y_combined_SVM_rbf = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_SVM_rbf))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_SVM_rbf = svm_rbf.predict(X_combined_std_SVM_rbf)
print('Misclassified combined samples: %d' % (y_combined_SVM_rbf != y_combined_pred_SVM_rbf).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_SVM_rbf, y_combined_pred_SVM_rbf))

print()

################################################################################
# Support Vector Machine
# ===== Best Parameter Tuned =====
# kernal - sigmoid
# C as the penalty parameter: 2.0
# gamma: 0.05
################################################################################
svm_rbf = SVC(kernel='sigmoid', gamma=0.05, C=2.0, random_state=0, max_iter = -1)
svm_rbf.fit(X_train_std, y_train)                      # do the training
print("================== Support Vector Machine (sigmoid) =================")
y_pred_SVM_rbf = svm_rbf.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_SVM_rbf).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_SVM_rbf))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_SVM_rbf = np.vstack((X_train_std, X_test_std))
y_combined_SVM_rbf = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_SVM_rbf))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_SVM_rbf = svm_rbf.predict(X_combined_std_SVM_rbf)
print('Misclassified combined samples: %d' % (y_combined_SVM_rbf != y_combined_pred_SVM_rbf).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_SVM_rbf, y_combined_pred_SVM_rbf))

print()

################################################################################
# Decision Tree (DT)
# ===== Best Parameter Tuned =====
# criterion: entropy
# max depth: 4
# min samples split: 0.5
# min samples leaf: 0.4
################################################################################
tree_DT = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=0.5, min_samples_leaf=0.4, random_state=0)
tree_DT.fit(X_train_std,y_train)
print("=========================== Decision Tree ===========================")
y_pred_DT = tree_DT.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_DT).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_DT))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_DT = np.vstack((X_train_std, X_test_std))
y_combined_DT = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_DT))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_DT = svm_rbf.predict(X_combined_std_DT)
print('Misclassified combined samples: %d' % (y_combined_DT != y_combined_pred_DT).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_DT, y_combined_pred_DT))

print()

################################################################################
# Random Forest (RF)
# ===== Best Parameter Tuned =====
# criterion: entropy
# number of estimator: 25
# min samples split: 2
# min samples leaf: 1
# number of jobs: 1
################################################################################
forest = RandomForestClassifier(criterion='entropy', n_estimators=25, min_samples_split=2, min_samples_leaf=1
                                ,random_state=0, n_jobs=1)
forest.fit(X_train_std,y_train)
print("=========================== Random Forest ===========================")
y_pred_RF = forest.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_RF).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_RF))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_RF = np.vstack((X_train_std, X_test_std))
y_combined_RF = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_RF))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_RF = forest.predict(X_combined_std_RF)
print('Misclassified combined samples: %d' % (y_combined_RF != y_combined_pred_RF).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_RF, y_combined_pred_RF))

print()

################################################################################
# KNeighbors (KNN)
# ===== Best Parameter Tuned =====
# number of neighbors: 18
# p: 2
# leaf size: 2
# metric: minkowski (euclidean)
################################################################################
knn = KNeighborsClassifier(n_neighbors=18, p=2, leaf_size=2, metric='minkowski')
knn.fit(X_train_std,y_train)
print("============================= KNeighbors =============================")
y_pred_KNN = knn.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred_KNN).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred_KNN))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std_KNN = np.vstack((X_train_std, X_test_std))
y_combined_KNN = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined_KNN))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred_KNN = knn.predict(X_combined_std_KNN)
print('Misclassified combined samples: %d' % (y_combined_KNN != y_combined_pred_KNN).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined_KNN, y_combined_pred_KNN))
