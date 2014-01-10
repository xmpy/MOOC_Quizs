#-*- coding:utf-8 -*-
##
## Machine Learning Foundation
## Quiz 2
## Positive And Negative Rays
##
## @author: zhaoxm(xmpy)
## 

import numpy as np
import unittest
import random
import math

##input:sample size
##output:[(-0.5,-1),(0.5,1)...]
def gen_sample_data(n):
	result_list = []
	for i in xrange(n):
		sample_x = 2*random.random() - 1
		sample_y = int(math.copysign(1,sample_x)) if random.random() < 0.8 else int(-math.copysign(1,sample_x))
		result_list.append((sample_x,sample_y))
	return result_list

##input:data
##output: ein
def get_ein(sample_data, s, theta):
	sample_size = len(sample_data)
	err_count = 0
	for sample_x, sample_y in sample_data:
		y = s * math.copysign(1,(sample_x - theta))
		if int(y) != sample_y:
			err_count +=1
	return float(err_count)/sample_size

##input:data
##outuput:parameter of model, theta, 
def find_best_para(sample_data):
	ein = 1
	theta_result= 0
	s_result = 0 

	for s in [-1,1]:
		for theta, unused in sample_data:
			ein_cand = get_ein(sample_data,s,theta)
			if ein_cand < ein:
				ein = ein_cand
				s_result,theta_result = s,theta
	return s_result,theta_result,ein

def get_eout(s, theta):
	eout = 0.5 + 0.3*s*(abs(theta)-1)
	return eout

def train_multi_model(train_data):
	target = train_data[:,-1]
	bst_ein = 1
	bst_theta = 0
	bst_s = 0
	bst_i = 0
	for i in xrange(train_data.shape[1]-1):
		sample_data = zip(train_data[:,i],target)
		s_result,theta_result,ein = find_best_para(sample_data)
		if ein < bst_ein:
			bst_ein = ein
			bst_s,bst_theta = s_result,theta_result
			bst_i = i
	return bst_s,bst_theta,bst_ein,bst_i


if __name__ == '__main__':
	### To run unittest
	#unittest.main()

	### For Question 17 & 18
	ein_sum = 0
	eout_sum = 0
	for i in xrange(5000):
		s_result,theta_result,ein = find_best_para(gen_sample_data(20))
		ein_sum += ein
		eout_sum += get_eout(s_result,theta_result)
	print "average Ein is:"
	print ein_sum/5000
	print "average Eout is:"
	print eout_sum/5000
	
	### For Question 19
	train_data = np.loadtxt('hw2_19_train.dat')
	bst_s,bst_theta,bst_ein,bst_i = train_multi_model(train_data)
	print bst_ein

	### For Question 20
	test_data = np.loadtxt('hw2_19_test.dat')
	target = train_data[:,-1]
	x = test_data[:,bst_i]
	sample_data = zip(x,target)
	print get_ein(sample_data, bst_s, bst_theta)