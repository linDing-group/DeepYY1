# import genInteraction_new
# import genNegativeData
# import genLabelData
# import genVecs
# import train
# import warnings
# import sys
# import os
# import CTCF
# from optparse import OptionParser


# def parse_args():
# 	parser = OptionParser(usage="CTCF Interaction Prediction", add_help_option=False)
# 	parser.add_option("-f", "--feature", default=100, help="Set the number of features of Word2Vec model")
# 	parser.add_option("-w","--word",default = 6)
# 	parser.add_option("-r","--range",default = 250)
# 	parser.add_option("-c","--cell",default = 'gm12878')
# 	parser.add_option("-t","--total",default = False)
# 	parser.add_option("-d","--direction",default = 'conv')


# 	(opts, args) = parser.parse_args()
# 	return opts

# def makepath(cell,direction):
# 	if os.path.exists("../Temp/%s" %(cell)) == False:
# 		os.makedirs("../Temp/%s" %(cell))
# 	if os.path.exists("../Temp/%s/%s" %(cell,direction)) == False:
# 		os.makedirs("../Temp/%s/%s" %(cell,direction))

# def run(word,feature,range,cell,total,direction):
# 	warnings.filterwarnings("ignore")
# 	makepath(cell,direction)
# 	if total!=False:
# 		print "Dealing with CTCF Motif Databese"
# 		CTCF.run(cell)

# 	if not os.path.isfile("../Temp/%s/CTCF.csv" %(cell)):
# 		print "Dealing with CTCF Motif Databese"
# 		CTCF.run(cell)

# 	if not os.path.isfile("../Temp/%s/CH.csv" %(cell)):
# 		print "Mapping Motifs to CHIA-PET"
# 		genInteraction_new.run(cell)

# 	if not os.path.isfile("../Temp/%s/%s/Negative.csv"%(cell,direction)):
# 		print "Generating Negative Data"		
# 		genNegativeData.run(cell,direction)
	
# 	if not os.path.isfile("../Temp/%s/%s/LabelData.csv"%(cell,direction)):
# 		genLabelData.run(cell,direction)

# 	if not os.path.isfile("../Temp/%s/Unsupervised"%(cell)):
# 		genVecs.Unsupervised(int(range),int(word),cell)
# 	if not os.path.isfile("../Temp/%s/%s/LabelSeq"%(cell,direction)):
# 		genVecs.gen_Seq(int(range),cell,direction)
# 	if not os.path.isfile("../Temp/%s/%s/datavecs.npy"%(cell,direction)):
# 		genVecs.run(word,feature,cell,direction)
	
# 	train.run(word,feature,cell,direction)
	
# def main():
# 	opts = parse_args()
# 	run(opts.word,opts.feature,opts.range,opts.cell,opts.total,opts.direction)

# if __name__ == '__main__':
# 	main()
#############################################################################################
#from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
import pandas as pd
import numpy as np
import os
import sys
import math
import random


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import warnings
from sklearn import preprocessing
import sklearn.preprocessing
from gensim import corpora, models, similarities

from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Reshape, normalization
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras.layers.recurrent import LSTM
from sklearn import metrics

def getWord_model(word,num_features,min_count, model, Unfile):
	word_model = ""
	if not os.path.isfile(model):
		sentence = LineSentence(Unfile,max_sentence_length = 15000)

		num_features = int(num_features)
		min_word_count = int(min_count)
		num_workers = 20
		context = 20
		downsampling = 1e-3

		print ("Training Word2Vec model...")
		word_model = Word2Vec(sentence, workers=num_workers,\
						size=num_features, min_count=min_word_count, \
						window=context, sample=downsampling, seed=1,iter = 50)
		word_model.init_sims(replace=False)
		word_model.save(model)

	else:
		print ("Loading Word2Vec model...")
		word_model = Word2Vec.load(model)

	return word_model

def DNAToWord(dna, K):

	sentence = ""
	length = len(dna)

	for i in range(length - K + 1):
		sentence += dna[i: i + K] + " "

	sentence = sentence[0 : len(sentence) - 1]
	return sentence


def getDNA_split(DNAdata,word):

	list1 = []
	list2 = []
	for DNA in DNAdata["seq1"]:
		DNA = str(DNA).upper()
		list1.append(DNAToWord(DNA,word).split(" "))#[['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC']]

	for DNA in DNAdata["seq2"]:
		DNA = str(DNA).upper()
		list2.append(DNAToWord(DNA,word).split(" "))

	return list1,list2

def getAvgFeatureVecs(DNAdata1,DNAdata2,model,num_features):
	counter = 0
	DNAFeatureVecs = np.zeros((len(DNAdata1),2*num_features), dtype="float32")
	
	for DNA in DNAdata1:
		if counter % 1000 == 0:
			print ("DNA %d of %d\r" % (counter, len(DNAdata1)))
			sys.stdout.flush()

		DNAFeatureVecs[counter][0:num_features] = np.mean(model[DNA],axis = 0)
		counter += 1
	print()
	
	counter = 0
	for DNA in DNAdata2:
		if counter % 1000 == 0:
			print ("DNA %d of %d\r" % (counter, len(DNAdata2)))
			sys.stdout.flush()
		DNAFeatureVecs[counter][num_features:2*num_features] = np.mean(model[DNA],axis = 0)
		counter += 1

	return DNAFeatureVecs

def npyTosvm(npyfile, svmfile, pos_num):
	dataDataVecs = np.load(npyfile)
	g = open(svmfile,'w')
	print(len(dataDataVecs))
	#print(dataDataVecs[0])
	m = 0
	for i in range(len(dataDataVecs)):
		line = ''
		for j in range(len(dataDataVecs[0])):
			if j == len(dataDataVecs[0])-1:
				line += str(j+1)+':'+str(dataDataVecs[i][j])+'\n'
			else:
				line += str(j+1)+':'+str(dataDataVecs[i][j])+'\t'
		m += 1
		if m < (pos_num+1):
			g.write('1\t'+line)
		else:
			g.write('0\t'+line)

def SVMtoCSV(svmfile, csvfile):
	f = open(svmfile,'r')
	g = open(csvfile,'w')
	lines = f.readlines()
	legth = len(lines[0].split('	'))-1
	#print(legth)
	classline = 'class'
	for i in range(legth):
		classline += ',%d'%(i+1)
	g.write(classline+'\n')

	for line in lines:
		line = line.strip('\n').split('	')
		g.write(line[0]+',')

		legth2 = len(line[1:])
		m = 0
		for j in line[1:]:
			if m == legth2-1:
				j = j.split(':')[-1]
				g.write(j)
				m += 1
			else:
				j = j.split(':')[-1]
				g.write(j+',')
				m += 1
		g.write('\n')

	f.close()
	g.close()

def precision(y_true, y_pred):
	# Calculates the precision
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	# Calculates the recall
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def f1(test_Y, pre_test_y):
	"""F1-score"""
	Precision = precision(test_Y, pre_test_y)
	Recall = recall(test_Y, pre_test_y)
	f1 = 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))
	return f1 

def TP(test_Y,pre_test_y):
	TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
	return TP

def FN(test_Y,pre_test_y):
	TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
	P=K.sum(K.round(K.clip(test_Y, 0, 1)))
	FN = P-TP #FN=P-TP
	return FN

def TN(test_Y,pre_test_y):
	TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
	return TN

def FP(test_Y,pre_test_y):
	N = (-1)*K.sum(K.round(K.clip(test_Y-K.ones_like(test_Y), -1, 0)))#N
	TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
	FP=N-TN
	return FP

def dnn_model(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size, CNNmodel):
	train_X = np.expand_dims(train_X, 2)
	test_X = np.expand_dims(test_X, 2)
	inputs = Input(shape = (train_X.shape[1], train_X.shape[2]))
	x = Conv1D(32, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(inputs)
	x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
	x = Flatten()(x)
	x = Dropout(0.5)(x)
	x = Dense(32, activation = 'relu')(x)
	x = Dense(16, activation = 'relu')(x)
	x = Dense(8, activation = 'relu')(x)
	predictions = Dense(1, activation = 'sigmoid')(x)
	model = Model(inputs = inputs, outputs = predictions)
	print("model")
	model.compile(optimizer = 'RMSprop',
				loss = 'mean_squared_error',
				metrics = ['acc',precision,recall,f1,TP,FN,TN,FP])
	print("compile")
	model.fit(train_X, train_Y, epochs = epoch, batch_size = 32, validation_data = (test_X, test_Y), shuffle = True)
	model.save(CNNmodel)
	pre_test_y = model.predict(test_X, batch_size = 50)
	pre_train_y = model.predict(train_X, batch_size = 50)
	test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
	train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
	print("train_auc: ", train_auc)
	print("test_auc: ", test_auc) 
	return test_auc

k = 2
Unfile = '%dUn'%(k)
g = open(Unfile,'w')

DNAseq = pd.read_csv('test1.fa',sep = "\t",error_bad_lines=False)
words1,words2 = getDNA_split(DNAseq,k)

for i in range(len(words1)):
	line = ' '.join(words1[i])
	g.write(line+'\n')

for i in range(len(words2)):
	line = ' '.join(words2[i])
	g.write(line+'\n')
g.close()

#get word2vec model
model = 'model_%d'%(k)
fea_num = 10
min_fea = 5
getWord_model(k,fea_num,min_fea,model,Unfile)

#obtain word2vec feature set

word_model = Word2Vec.load(model)
dataDataVecs = getAvgFeatureVecs(words1,words2,word_model,fea_num)
print (dataDataVecs.shape)
fea_npy = '%d_vecs.npy'%(k)
np.save(fea_npy,dataDataVecs)


# npy To csv and 
fea_svm = '%d_vecs.svm'%(k)
fea_csv = '%d_vecs.csv'%(k)
pos_number = 4
npyTosvm(fea_npy, fea_svm,pos_number)
SVMtoCSV(fea_svm, fea_csv)


##############################CNN

data = np.array(pd.read_csv(fea_csv))
X1 = data[0:pos_number, 1:]
Y1 = data[0:pos_number, 0]
X2 = data[pos_number:, 1:]
Y2 = data[pos_number:, 0]
X = np.concatenate([X1, X2], 0)
Y = np.concatenate([Y1, Y2], 0)
#Y = Y.reshape((Y.shape[0], -1))
print (X)
print ("X.shape: ", X.shape)
print ("Y.shape: ", Y.shape)

lr = 0.4
epoch = 20
batch_size = 32
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
#kf = KFold(n_splits = 5, shuffle = False)
kf = kf.split(X)
cnn_model = '%d.h5'%(k)
result = '%d_cnn_result.txt'%(k)

test_aucs = []
for i, (train_fold, validate_fold) in enumerate(kf):
    print("\n\ni: ", i)
    test_auc = dnn_model(X[train_fold], Y[train_fold], X[validate_fold], Y[validate_fold], lr, epoch, batch_size,cnn_model)
    test_aucs.append(test_auc)
w = open(result, "w")
for j in test_aucs: 
    w.write(str(j) + ',')
w.write('\n')
w.write(str(np.mean(test_aucs)) + '\n')
w.close()