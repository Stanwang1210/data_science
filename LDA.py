#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:04:19 2021

@author: wangshihang
"""

import numpy as np
import sys

data = [(5,3, 0), (3,5, 0), (3, 4, 0), (4,5, 0), (4, 7, 0), (5,6, 0)
        ,(9, 10, 1), (7, 7, 1), (8, 5, 1), (8, 8, 1), (7, 2, 1), (10, 8, 1)]

sample = []
class1 = []
class2 = []
for d in data:
    point = [d[0], d[1]]
    sample.append(point)
    if d[2] == 0:
        class1.append(point)
    else:
        class2.append(point)
 
# data = np.concatenate(class1, class2, )   
print("Sample variance is ", np.var(sample, ddof = 1))
class1= np.array(class1)
class2= np.array(class2)
u1 = np.mean(class1, axis = 0).reshape(2, 1)
u2 = np.mean(class2, axis = 0).reshape(2, 1)

SB = np.matmul((u1 - u2), (u1 - u2).T) 

S1 = np.zeros((2,2))
S2 = np.zeros((2,2))
# print(np.shape(class1[0]))
for point in class1:
    S1 += np.matmul((point.reshape(2, 1)-u1), (point.reshape(2, 1)-u1).T)
for point in class2:
    S2 += np.matmul((point.reshape(2, 1)-u2), (point.reshape(2, 1)-u2).T)
    
SW = S1 + S2

A = np.matmul(np.linalg.inv(SW), SB)

eigen_values, eigen_vectors = np.linalg.eig(A)
print("Max Eigen valud ", eigen_values[0]  )
# print(eigen_vectors)
print("Normalize Eigen vector ", eigen_vectors[0] / np.linalg.norm(eigen_vectors[0]))
# print(new_data.shape)
