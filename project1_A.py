###############################################################################
# FILE: project1_A.py
# AUTHOR: Jaehyuk Choi (1215326372)
# CONTACT INFO: jchoi154@asu.edu
#
# COURSE INFO
# EEE591 Fall 2019
# Homework5 10-10-2019 / Time: 9:40am - 10:30am MWF
#
# DESCRIPTION
# The project is to practice of analyzing data and predicting an output from
# medical sample data by using machine learning techniques. The goal of the
# project is to predict heart disease from the data provided by Acme Medical
# Analysis and Prediction Enterprises (AMAPE).
#
# The given input data is heart1.csv which contains 270 samples
# aligned for 13 features and 1 actual result such as 1 if absence of heart
# disease or 2 if presence of that, based on those 13 features.
#
# THe program reads the raw data and outputs a correlated list, a covariance
# heat plot, a pair plot, a correlation matrix, and a covariance matrix to
# fascilitate finding how strongly variables (features) are correlated each other.
###############################################################################

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # used for plotting
from matplotlib import cm as cm         # for the color map
import seaborn as sns                   # data visualization

################################################################################
# Function to create covariance for dataframes                                 #
# Inputs:                                                                      #
#    mydataframe - the data frame to analyze                                   #
#    numtoreport - the number of highly correlated pairs to report             #
# Outputs:                                                                     #
#    correlations are printed to the screen                                    #
################################################################################

def mosthighlycorrelated(mydataframe, numtoreport):
    cormatrix = mydataframe.corr()                      # find the correlations

    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones.
    # (The diagonal is always 1; the matrix is symmetric about the diagonal.)

    # shape returns a tuple, so the * in front of the expression allows the
    # tri function to unpack the tuple into two separate values: rows and cols.

    # The tri function creates an array filled with 1s in the shape of a
    # triangle. If k is 0, then the diagonal and below are all 1s, the rest 0s.
    # If k=-1, the diagonal is 0 but all below the diagonal are 1. If k=-2,
    # then the entries below the diagonal are also 0, and so on. Finally, the
    # transpose is done so that we only keep values above the diagonal.
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T

    # find the top n correlations
    cormatrix = cormatrix.stack()     # rearrange so the reindex will work...

    # Reorder the entries so they go from largest at top to smallest at bottom
    # based on absolute value
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()

    # assign human-friendly names
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    print("\nMost Highly Correlated")
    print(cormatrix.head(numtoreport))     # print the top values

################################################################################
# Function to create the Correlation matrix                                    #
# Input:                                                                       #
#    X - a dataframe                                                           #
# Output:                                                                      #
#    The correlation matrix plot                                               #
################################################################################
def correl_matrix(X):
    # create a figure that's 7x7 (inches?) with 100 dots per inch
    fig = plt.figure(figsize=(7,7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet',30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0,len(X.columns),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(X.columns,fontsize=9)
    ax1.set_yticklabels(X.columns,fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()

################################################################################
# Function to create the pair plots                                            #
# Input:                                                                       #
#    df - a dataframe                                                          #
# Output:                                                                      #
#    The pair plots                                                            #
################################################################################
def pairplotting(df):
    sns.set(style='whitegrid', context='notebook')   # set the apearance
    sns.pairplot(df,height=2.5)                      # create the pair plots
    plt.show()                                       # and show them

pd.set_option('display.max_columns', 14)             # shows all the columns of
                                                     # of the correlation and
                                                     # the covariance matrices
# this creates a dataframe similar to a dictionary
# a data frame can be constructed from a dictionary
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

heart = pd.read_csv('heart1.csv')                    # reads the raw data.
#print('first 5 observations',heart.head(5))
cols = heart.columns

#  descriptive statistics
#print('\nDescriptive Statistics')
#print(heart.describe())

mosthighlycorrelated(heart,14)               # generates correlated list in descending order
correl_matrix(heart)                         # generates the covariance heat plot
pairplotting(heart)                          # generates the pair plot

# prints correlation matrix
print("============================= Correlation Matrix =============================")
correl_matrix = heart.corr()
print(correl_matrix)

print()

# prints covariance matrix
print("============================== Covariance Matrix ==============================")
names = list(correl_matrix.iloc[:0])
matrix_heart = np.array(heart).T
covar_matrix = np.cov(matrix_heart)
covar_matrix_pd = pd.DataFrame(covar_matrix, columns=names, index=names)
print(covar_matrix_pd)
