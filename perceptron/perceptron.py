"""
This file contrains all the function needed for perceptron homework.
They are all written in Python 2.
"""
import numpy as np

def split(read_file, train_file, val_file, num_train = 4000, num_val = 1000):
	"""
	Splits dataset in a file into 2 files.
	"""

	with open(read_file,'r') as txt:
		lines = txt.readlines()

	train_lines = lines[:num_train]
	val_lines = lines[num_train:]

	with open(train_file, 'w') as train:
		train.writelines(train_lines)

	with open(val_file, 'w') as val:
		val.writelines(val_lines)



def words(file, X, N=4000):
	"""
	words(data, X) takes data as input and returns a Python list 
	containing all the words that occur in at least X emails
	"""
	num_train = N
	with open(file, "r") as traintxt:
		word_dict = {}
		for i in range(num_train):
			line = traintxt.readline()
			word_list = line.split()
			spam = word_list.pop(0)
			word_set = set(word_list)

			for word in word_set:
				try:
					word_dict[word] += 1
				except KeyError:
					word_dict[word] = 1

	final_word_list = []
	for word in word_dict.keys():
		if word_dict[word]>=X:
			final_word_list.append(word)
	return final_word_list

def feature_vector(email,WORDS):
	"""
	feature_vector(email) takes as input a single email and returns 
	the corresponding feature vector as a Python list.
	"""
	word_list = email.split()
	spam = int(word_list.pop(0))
	if spam==0:
		spam=-1
	total_words = len(WORDS)
	feature_vector = np.zeros(total_words)
	for i in range(total_words):
		if WORDS[i] in word_list:
			feature_vector[i] = 1
	feature_vector_list = feature_vector.tolist()
	return feature_vector_list, spam

def save_feature_vector(WORDS, read_file, write_file=None, name = None, N=None, verbose=True):
	"""
	Read dataset of emails and convert them into feature vectors.
	Then save feature vectors in write_file.
	"""
	training_data = []
	printevery=100
	with open(read_file, "r") as traintxt:
	    if verbose and name:
	    	print name
	    train_lines = traintxt.readlines()
	    if N is None:
	    	N = len(train_lines)
	    for i in range(N):
	    	email = train_lines[i]
	        vector, label = feature_vector(email,WORDS)
	        new_vector = [label] + vector
	        training_data.append(new_vector)
	        if verbose and i%printevery==0:
	            print "Processed %d emails." % (i,)
	if verbose and name:
		print "Finished processing " + name + "."
	np_train = np.array(training_data)
	if write_file:
		np.savetxt(write_file, np_train, delimiter=',',fmt="%1d")

def load_feature_vector(file, size=None):
	"""
	Load feature vectors from txt file.
	"""
	feature_vector = []
	labels = []
	with open(file,"r") as txt:
		lines = txt.readlines()
		length = size if size else len(lines)
		for i in range(length):
			line = lines[i]
			features = line.split(",")
			label = int(features.pop(0))
			feature_vector.append(features)
			labels.append(label)
	feature_vector = np.array(feature_vector,dtype=np.float)
	labels = np.array(labels)
	return feature_vector, labels

def perceptron_train(X, y, verbose=False, max_pass = 100):
	"""
	Takes feature vectors of all training set in numpy array form.
	perceptron_train(X, y) trains a perceptron classifier  
	Inputs:
	- X: inputs
	- y: labels
	Outputs:
	- W: final classification vector
	- k: number of updates performed
	- iter: number of passes through the data
	"""
	M, N = X.shape
	W = np.zeros((N))
	iter = 0
	k = 0
	print_every = 1
	kk = 0 # Count for single pass
	error = 1
	while error>0 and iter<max_pass:
		iter += 1
		for i in range(M):
			if y[i] * W.dot(X[i].T)<=0 and np.sum(X[i])!=0:
				W += y[i]*X[i]
				k += 1
				kk += 1
		error = float(kk) / M
		kk = 0 
		if verbose and iter%print_every==0:
			print "Number of iteration: %d, Accuracy: %f" % (iter,1-error)

	return W, k, iter


def perceptron_error(W, X, y):
	"""
	Returns the error rate of the model with weight W with input X and label y.
	"""
	prediction = W.dot(X.T)
	prediction[prediction>0] = 1
	prediction[prediction<0] = -1
	correct = prediction==y
	accuracy = np.mean(correct)
	return 1-accuracy

