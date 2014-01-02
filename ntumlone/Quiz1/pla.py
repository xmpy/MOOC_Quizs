#-*- coding:utf-8 -*-
##
## Machine Learning Foundation
## Quiz 1
##
## @author: zhaoxm(xmpy)
## 

import numpy as np
import unittest

# Find the first incorrect sample since start_index
def find_next_incorrect(in_arr, weight, start_index):
	i = start_index
	r = 0
	while True:
		row = in_arr[i%in_arr.shape[0]]
		if not ((np.dot(weight,row[:-1]) > 0 and row[-1] == 1) or (np.dot(weight,row[:-1]) <= 0 and row[-1] == -1)):
			r = i%in_arr.shape[0]
			break
		i += 1
	return r

def run_pla_native_cycle(in_arr):
	#initialize the weight as zero 
	weight = np.zeros(in_arr.shape[1])

	#initial the learning rate
	mu = 1

	# add the column 1
	cols_1 = np.ones((in_arr.shape[0],1))
	in_arr = np.hstack((cols_1, in_arr))

	correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
	count = 0
	start_index = 0

	while incorrect_arr.any():
		count += 1
		#pick_one_arr = incorrect_arr[np.random.randint(0,incorrect_arr.size)]
		start_index = find_next_incorrect(in_arr, weight, start_index)
		pick_one_arr = in_arr[start_index]
		weight += mu * pick_one_arr[-1] * (pick_one_arr[:-1])
		correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
	return weight,count

###	Based on the weight, split input array into correct one and incorrect one 
def split_correct_incorrect(in_arr, weight):
	_correct_arr = np.array([])
	_incorrect_arr = np.array([])

	for row in in_arr:
		if (np.dot(weight,row[:-1]) > 0 and row[-1] == 1) or (np.dot(weight,row[:-1]) <= 0 and row[-1] == -1):
			if not _correct_arr.any():
				_correct_arr = np.array([row.copy()])
			else:
				_correct_arr = np.vstack((_correct_arr,row.copy()))
		else:
			if not _incorrect_arr.any():
				_incorrect_arr = np.array([row.copy()])
			else:
				_incorrect_arr = np.vstack((_incorrect_arr,row.copy()))
	return (_correct_arr, _incorrect_arr)

def run_pla(in_arr,mu):
	#initialize the weight as random numbers
	weight = np.random.random((1,in_arr.shape[1])) 

	# add the column 1
	cols_1 = np.ones((in_arr.shape[0],1))
	in_arr = np.hstack((cols_1, in_arr))

	correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
	count = 0

	while incorrect_arr.any():
		count += 1
		#pick_one_arr = incorrect_arr[np.random.randint(0,incorrect_arr.size)]
		pick_one_arr = incorrect_arr[np.random.randint(0,incorrect_arr.shape[0])]
		weight += mu * pick_one_arr[-1] * (pick_one_arr[:-1])
		correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
	return weight,count

def run_pocket_pla(in_arr, iter_count):
	#initialize the weight as random number between 0 - 10
	#weight = np.zeros(in_arr.shape[1])
	weight = np.random.random((1,in_arr.shape[1])) 
	weight_cand = weight.copy()

	#initial the learning rate
	mu = 1

	# add the column 1
	cols_1 = np.ones((in_arr.shape[0],1))
	in_arr = np.hstack((cols_1, in_arr))

	correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
	count = 0

	while incorrect_arr.any():
		count += 1
		correct_count = correct_arr.shape[0]
		#pick_one_arr = incorrect_arr[np.random.randint(0,incorrect_arr.size)]
		pick_one_arr = incorrect_arr[np.random.randint(0,incorrect_arr.shape[0])]
		weight += mu * pick_one_arr[-1] * (pick_one_arr[:-1])
		correct_arr,incorrect_arr = split_correct_incorrect(in_arr,weight)
		if correct_arr.shape[0] > correct_count:
			weight_cand = weight.copy()
		if count >= iter_count:
			break
	return weight_cand,count

#### For Test
class PLATestCase(unittest.TestCase):
	def testSplitCorrectIncorrect(self):
		test_arr = np.array([[1,3,1],[1,2,-1]])
		correct_arr, incorrect_arr = split_correct_incorrect(test_arr,np.array([1,1]))
		self.assertAlmostEqual(np.array_equal(correct_arr, np.array([[1,3,1]])), True)
		self.assertAlmostEqual(np.array_equal(incorrect_arr, np.array([[1,2,-1]])), True)
		correct_arr, incorrect_arr = split_correct_incorrect(test_arr,np.array([-1,-1]))
		self.assertAlmostEqual(np.array_equal(incorrect_arr, np.array([[1,3,1]])), True)
		self.assertAlmostEqual(np.array_equal(correct_arr, np.array([[1,2,-1]])), True)
	
	def testRunPlaNativeCycle(self):
		#test_arr = np.array([[1,3,1],[1,2,-1]])
		test_arr = np.loadtxt('test.dat')
		print run_pla_native_cycle(test_arr)

	def testRunPla(self):
		#test_arr = np.array([[1,3,1],[1,2,-1]])
		test_arr = np.loadtxt('test.dat')
		print run_pla(test_arr)

	def testRunPla2000(self):
		test_arr = np.loadtxt('test.dat')
		count_sum = 0
		for i in xrange(100):
			weight, count = run_pla(test_arr)
			count_sum += count
		print count_sum / 100

	def testRunPocketPla(self):
		train_arr = np.loadtxt('hw1_18_train.dat')
		print train_arr.shape
		test_arr = np.loadtxt('hw1_18_test.dat')
		cols_1 = np.ones((test_arr.shape[0],1))
		test_arr = np.hstack((cols_1, test_arr))
		correct_per = 0

		for i in xrange(100):
			weight,count = run_pocket_pla(train_arr)
			correct_arr,incorrect_arr = split_correct_incorrect(test_arr,weight)
			correct_per += float(correct_arr.shape[0]) / test_arr.shape[0]
		print correct_per/100

if __name__ == '__main__':
	### To run unittest
	#unittest.main()

	### For Question 15
	test_arr = np.loadtxt('hw1_15.dat')
	print "Answer for Question 15"
	print "weigt, updates count:",run_pla_native_cycle(test_arr)
	print "\n"

	### For Question 16
	### 2000 times spent too much, so I only repeat 100times
	count_sum = 0
	for i in xrange(100):
		weight, count = run_pla(test_arr,1)
		count_sum += count
	print "Answer for Question 16"
	print "Average Updates:",count_sum/100
	print "\n"

	### For Question 17
	### 2000 times spent too much, so I only repeat 100times
	count_sum = 0
	for i in xrange(100):
		weight, count = run_pla(test_arr,0.5)
		count_sum += count
	print "Answer for Question 17"
	print "Average Updates:",count_sum/100
	print "\n"

	### For Question 18
	### 2000 times spent too much, so I only repeat 100times
	train_arr = np.loadtxt('hw1_18_train.dat')
	test_arr = np.loadtxt('hw1_18_test.dat')

	cols_1 = np.ones((test_arr.shape[0],1))
	test_arr = np.hstack((cols_1, test_arr))
	incorrect_per = 0
	
	for i in xrange(100):
		weight,count = run_pocket_pla(train_arr,50)
		correct_arr,incorrect_arr = split_correct_incorrect(test_arr,weight)
		incorrect_per += 1 - float(correct_arr.shape[0]) / test_arr.shape[0]
	print "Answer for Question 18"
	print "Average Error Rate:",incorrect_per/100
	print "\n"

	for i in xrange(100):
		weight,count = run_pocket_pla(train_arr,100)
		correct_arr,incorrect_arr = split_correct_incorrect(test_arr,weight)
		incorrect_per += 1 - float(correct_arr.shape[0]) / test_arr.shape[0]
	print "Answer for Question 20"
	print "Average Error Rate:",incorrect_per/100
	print "\n"
