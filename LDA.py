#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:04:19 2021

@author: wangshihang
"""

from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys

data = [(5,3, 0), (3,5, 0), (3, 4, 0), (4,5, 0), (4, 7, 0), (5,6, 0)
        ,(9, 10, 1), (7, 7, 1), (8, 5, 1), (8, 8, 1), (7, 2, 1), (10, 8, 1)]

new_data = np.array(data)[:, :-1]
# print(new_data.shape)
print("Sample varince is ", np.var(new_data, ddof=1, axis=0))

# sys.exit(0)
# for i in range(len(data)):
    
# wine = load_wine()
df = pd.DataFrame(data, columns=["x", "y", "class"])
# y = pd.Categorical.from_codes(wine.target, wine.target_names)
# print(y)
# df = X.join(pd.Series(y, name='class'))

class_feature_means = pd.DataFrame(columns=[0, 1])
# class_feature_means.drop(class_feature_means.tail(1).index,inplace = True)
for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()

class_feature_means = class_feature_means.iloc[:-1, ]
within_class_scatter_matrix = np.zeros((2,2))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    
    s = np.zeros((2,2))
    for index, row in rows.iterrows():
        # print(row.shape)
        # print(class_feature_means[c].shape)
        # sys.exit(0)
        x, mc = row.values.reshape(2,1), class_feature_means[c].values.reshape(2,1)
        
        s += (x - mc).dot((x - mc).T)
    
        within_class_scatter_matrix += s

feature_means = df.iloc[:,:-1].mean()
between_class_scatter_matrix = np.zeros((2,2))
for c in class_feature_means:    
    n = len(df.loc[df['class'] == c].index)
    
    mc, m = class_feature_means[c].values.reshape(2,1), feature_means.values.reshape(2,1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))
print("Eigen value ", eigen_values)
# print("Eigen vector ", eigen_vectors)
for eigen_vector in eigen_vectors:
    print("Eigen vector ",eigen_vector / np.linalg.norm(eigen_vector))
sys.exit(0)
pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])

eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))
