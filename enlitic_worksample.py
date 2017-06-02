# -*- coding: utf-8 -*-
"""
 Enlitic - Work sample on 

 Kernel density estimation with Mixture of Gaussians
 
 Coded by Sudhir Sornapudi
 Email: ssbw5@mst.edu
"""

"""
 Main file to train and test on MNIST and CIFAR100 datasets
"""



import time
import cPickle
import numpy as np
#from numpy.matlib import repmat
from matplotlib import pyplot as plt
import matplotlib.cm as cm
#from math import pi
#import multiprocessing as mp
from decimal import *
import csv
import sys
import argparse

import kdeGauss # import the designed kde model

getcontext().prec = 7 # Precision for decimal
"""
 Function to unpickle the given datasets
 returns: unpickled file
"""
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
"""    
 Fuction to load and preprocess MNIST dataset 
 returns: Preprocessed train, valid and test datasets from MNIST
"""
def load_data_mnist(data_dir):
    Data = unpickle(data_dir)
    seed = 3478954
    X_training = Data[0][0]
    np.random.seed(seed)
    np.random.shuffle(X_training)
    X_train = X_training[0:10000]
    X_valid = X_training[10000:20000]
    X_test = Data[2][0]
    return X_train, X_valid, X_test
"""    
 Fuction to load and preprocess MNIST dataset    
 returns: Preprocessed train, valid and test datasets from CIFAR100
"""
def load_data_cifar(train_dir, test_dir):
    #meta_data = './cifar-100-python/meta'
    Train = unpickle(train_dir)
    Test = unpickle(test_dir)
    seed = 3478954
    X_training =Train['data'].astype(np.float64)
    X_training = X_training/255
    np.random.seed(seed)
    np.random.shuffle(X_training)
    X_train = X_training[0:10000]
    X_valid = X_training[10000:20000]
    X_test =Test['data'].astype(np.float64)
    X_test = X_test/255
    return X_train, X_valid, X_test
"""
 Function to visualize the loaded dataset
 returns: plot data
"""
def visualize(X, n=20, data_img_rows=28, data_img_cols=28, data_img_channels=1):
    N = n**2
    D = data_img_channels
    R = data_img_rows
    C = data_img_cols
    montage = X[0:N].reshape(N, D, R, C).reshape(n, n, D, R, C).transpose(0, 1, 3, 4, 2)
    img = montage.swapaxes(1, 2).reshape(R*n, C*n, D)
    if D == 1:
        img = img.reshape(R*n, C*n)
    fig = plt.imshow(img,cmap = cm.gray)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return plt
"""    
 Function to save the results in a comma separated file
 returns: None
""" 
def saveResults_csv(Filename = 'kde_results.csv'):
    csvfile = open(Filename, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['sigma','L_D_valid'])
    for sig,l in zip(sigma,L_valid): 
        writer.writerow([sig, l])
    writer.writerow([' '])
    writer.writerow(['optimal sigma', sigma_optimal])
    writer.writerow(['L_D_test', L_test])
    writer.writerow(['Run_time_test(in sec)', round(run_time,2)])
    csvfile.close()
    
#%%
"""
 Get the arguments from Linux shell or command prompt
"""
parser = argparse.ArgumentParser(prog='enlitic_worksample.py')
parser.add_argument('dataset_name', nargs = 1, help='"mnist" or "cifar"')
args = parser.parse_args()

try:
    dataset = str(sys.argv[1])
except:
    print('TODO: Please pass dataset_name : "mnist" or "cifar"')

# Choose MINST dataset or CIFAR100 dataset 
if (dataset == 'mnist' or dataset == 'MNIST'):
    X_train, X_valid, X_test = load_data_mnist(data_dir = './mnist/mnist.pkl')
    img = visualize(X_train, n=20, data_img_rows=28, data_img_cols=28, data_img_channels=1)
    img.savefig(dataset+'.png')
    
elif (dataset == 'cifar' or dataset == 'CIFAR' or dataset == 'cifar100' or dataset == 'CIFAR100' ):
    X_train, X_valid, X_test = load_data_cifar(train_dir = './cifar-100-python/train', test_dir = './cifar-100-python/test')
    img = visualize(X_train, n=20, data_img_rows=32, data_img_cols=32, data_img_channels=3)
    img.savefig(dataset+'.png')
	
else:
	print('TODO: Please pass dataset_name as arg: "mnist" or "cifar"')
	exit()
    
print("---- Working on {} dataset ----".format(dataset.upper()))

sigma = [0.05, 0.08, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00] # Grid search
L_valid = [] # list to store mean of log-likelihood values
for sg in sigma:
    print("Training with sigma = {}".format(sg))
    kde_prob = kdeGauss.model(X_train[0:10], X_valid[0:10], sg)
    print ("L_D_valid with sigma {} = {}".format(sg, kde_prob))
    L_valid.append(kde_prob)

max_idx = np.argmax(L_valid)
#Optimal sigma
sigma_optimal = sigma[max_idx]

start_test = time.time()
print ("----  Predicting model with optimal sigma ----")
print("Optimal sigma from training = {}".format(sigma_optimal))

L_test = kdeGauss.model(X_train[0:10],X_test[0:10],sigma_optimal)

print ("L_D_test with optimal sigma = {}".format(L_test))
run_time = time.time() - start_test
print("--- Run time on test data is {} seconds ---".format(run_time))

# Save results in a csv file
saveResults_csv(dataset+'_results.csv')