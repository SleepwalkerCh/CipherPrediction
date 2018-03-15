#coding=utf-8
import tensorflow as tf
import numpy as np
import Data
import os
from FindMaxNIndex import FindIndex
from copy import deepcopy
import random
class RNN:
	batch_size = 8  # one sample data, one batch
	max_sequence_length = 8
	def CreateNetwork(self):
		print("Creating Network.....",end=" ")
		tf.set_random_seed(777)  # reproducibility
		idx2char = Data.GetCharsSet()
		# hyper parameters
		hidden_size = 66  # RNN output size
		num_classes = len(idx2char)  # final output size (RNN or softmax, etc.)
		learning_rate = 0.01
		layer_num = 2

		self.X = tf.placeholder(tf.int32, [RNN.batch_size, RNN.max_sequence_length])  # X data
		self.Y = tf.placeholder(tf.int32, [RNN.batch_size, RNN.max_sequence_length])  # Y label

		x_one_hot = tf.one_hot(self.X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
		# LSTM cell
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
		# Dropout Layer
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
		# multi-LSTM cell
		mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

		init_state = mlstm_cell.zero_state(batch_size=RNN.batch_size, dtype=tf.float32)
		x_length = [1,2,3,4,5,6,7,8]
		outputs, _states = tf.nn.dynamic_rnn(mlstm_cell, inputs=x_one_hot,sequence_length=x_length,
											 initial_state=init_state, dtype=tf.float32, time_major=False)

		# FC layer
		X_for_fc = tf.reshape(outputs, [-1, hidden_size])
		outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
		#outputs = tf.nn.softmax(outputs,axis=1)
		# reshape out for sequence_loss
		self.outputs = tf.reshape(outputs, [RNN.batch_size, -1, num_classes])

		weights = tf.ones([RNN.batch_size, RNN.max_sequence_length])
		# http://blog.csdn.net/appleml/article/details/54017873
		sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.Y, weights=weights)
		self.loss = tf.reduce_mean(sequence_loss)
		self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
		print("OK")
	def Train(self,data,times): # data = [X,Y] X = [[10],[10],……,[10]] Y = [[10],[10],……,[10]]
		idx2char = Data.GetCharsSet()
		char2idx = {c: i for i, c in enumerate(idx2char)}

		with tf.Session() as self.sess:
			self.sess.run(tf.global_variables_initializer())
			for j in range(times):
				for i in range(len(data)):

					x_data, y_data = Data.Pwd2Batch(data[random.randrange(0,len(data))],RNN.max_sequence_length)
					x_idx = [[char2idx[c] for c in x_data[k]] for k in range(RNN.max_sequence_length)]
					y_idx = [[char2idx[c] for c in y_data[k]] for k in range(RNN.max_sequence_length)]
					#x_idx = np.reshape(x_idx, [-1, RNN.max_sequence_length])
					#y_idx = np.reshape(y_idx, [-1, RNN.max_sequence_length])
					l, _ = self.sess.run([self.loss, self.train],
									feed_dict={self.X: x_idx, self.Y: y_idx})

				if j % 100 == 0:
					self.TestPwdProb('zaishuan')
					self.TestPwdProb('zaishuam')
					self.TestPwdProb('zaishuap')
					print(j, "loss:", l)

			# self.TestPwdProb('0123456')
			# OutFile = open("Prediction.txt", 'w')
			# self.Predict(list("          "),OutFile=OutFile)
			# OutFile.close()
	def Predict(self,Last,depth=0,OutFile=None):
		if depth > 12:
			return
		else:
			depth = depth + 1
		idx2char = Data.GetCharsSet()
		char2idx = {c: i for i, c in enumerate(idx2char)}
		x_idx = [char2idx[c] for c in Last]
		x_idx = np.reshape(x_idx, [-1, 10])

		Probability = self.sess.run(self.outputs, feed_dict={self.X: x_idx})
		Probability = Probability[0][len(Probability[0]) - 1].tolist()

		temp = ['a' for i in range(10)]
		for i in range(9):
			temp[i] = Last[i + 1]
		# 针对最后一个字符进行递归
		width = 2
		for i in range(1,width + 1):
			temp[9] = idx2char[FindIndex(Probability, i)]
			if temp[9] == ' ':
				continue
			if temp[9] != 'E':
				self.Predict(deepcopy(temp),depth,OutFile)
			else:
				result = ''.join(temp)
				result = result.replace(" ", "")
				result = result[:result.index('E')]
				if OutFile is None:
					print(result)
				else:
					OutFile.write(str(result) + '\n')
	def softmax(self,x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def TestPwdProb(self, TestString):
		idx2char = Data.GetCharsSet()
		char2idx = {c: i for i, c in enumerate(idx2char)}
		x_data,y_data = Data.Pwd2Batch(TestString,RNN.max_sequence_length)
		x_idx = [[char2idx[c] for c in x_data[k]] for k in range(RNN.max_sequence_length)]
		result = 1

		Probability = self.sess.run(self.outputs, feed_dict={self.X: x_idx})
		Probability = Probability.tolist()
		for i in range(RNN.batch_size):
			for j in range(RNN.max_sequence_length):
				Probability[i][j] = self.softmax(Probability[i][j]).tolist()
		for i in range(RNN.batch_size):
			for j in range(RNN.max_sequence_length):
				print(idx2char[Probability[i][j].index(max(Probability[i][j]))],end='')
			print("",end=" ")
		for i in range(RNN.max_sequence_length - 1):
			result = result * Probability[i][i][char2idx[TestString[i + 1]]]
		print(result)

