#coding=utf-8
import pickle
from copy import deepcopy
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from Src.FindMaxNIndex import FindIndex
from Src.FirstCharProb import FirstCharProb

from Src.Data import Data


class RNN:
	batch_size = 8  # one sample Data, one batch
	max_sequence_length = 8
	tensorboard_route = "F:/TrainData/logs"
	is_new_train = False
	total_step = 0
	def LstmCell(self,hidden_size):
		# LSTM cell
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
		# Dropout Layer
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
		return lstm_cell
	def CreateNetwork(self):
		print("Creating Network.....")
		tf.set_random_seed(777)  # reproducibility
		idx2char = Data.GetCharsSet()
		# hyper parameters
		hidden_size = 50  # RNN output size
		num_classes = len(idx2char)  # final output size (RNN or softmax, etc.)
		learning_rate = 0.001
		layer_num = 1

		self.X = tf.placeholder(tf.int32, [RNN.batch_size, RNN.max_sequence_length],name="X")  # X Data
		self.Y = tf.placeholder(tf.int32, [RNN.batch_size, RNN.max_sequence_length],name="Y")  # Y label

		x_one_hot = tf.one_hot(self.X, num_classes,name="x_one_hot")  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
		# LSTM cell
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
		# Dropout Layer
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
		# multi-LSTM cell
		mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.LstmCell(hidden_size) for _ in range(layer_num)], state_is_tuple=True)

		init_state = mlstm_cell.zero_state(batch_size=RNN.batch_size, dtype=tf.float32)
		x_length = [1,2,3,4,5,6,7,8]
		outputs,_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x_one_hot,sequence_length=x_length,initial_state=init_state, dtype=tf.float32, time_major=False)
		# FC layer
		X_for_fc = tf.reshape(outputs, [-1, hidden_size],name="output")
		#print(X_for_fc.get_shape())
		outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

		# reshape out for sequence_loss
		self.outputs = tf.reshape(outputs, [RNN.batch_size, -1, num_classes])

		weights = tf.ones([RNN.batch_size, RNN.max_sequence_length])
		# http://blog.csdn.net/appleml/article/details/54017873
		# print(self.outputs.get_shape())
		# print(self.Y.get_shape())
		sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.Y, weights=weights)
		self.loss = tf.reduce_mean(sequence_loss)
		tf.summary.scalar('loss', self.loss)
		self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
		self.sess = tf.Session()
		print("OK")
	def SaveSteps(self):
		file = open('../Model/steps','wb')
		pickle.dump(RNN.total_step,file)
		file.close()
	def SaveModel(self):
		pb_file_path = '../Model/LSTM.pb'
		constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["output"])
		with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
			f.write(constant_graph.SerializeToString())
	def RestoreSteps(self):
		file = open('../Model/steps', 'rb')
		RNN.total_step = pickle.load(file)
		file.close()
	def Train(self):
		print("Training...")
		saver = tf.train.Saver(max_to_keep=1)
		merged_summary_op = tf.summary.merge_all()
		writer = tf.summary.FileWriter(RNN.tensorboard_route, self.sess.graph)
		if RNN.is_new_train == True:
			self.sess.run(tf.global_variables_initializer())
		else:
			saver.restore(sess=self.sess, save_path="../Model/model.ckpt")
			self.RestoreSteps()
		# Start
		for j in range(len(Data.batches)):
			x_idx,y_idx = Data.GetBatch()
			#print(len(x_idx))
			#print(len(y_idx))
			# print(y_idx.getshape())
			self.sess.run(self.train,feed_dict={self.X: x_idx, self.Y: y_idx})
			if j % 2000== 0:
				summary_str,loss = self.sess.run([merged_summary_op,self.loss]
									,feed_dict={self.X: x_idx, self.Y: y_idx})
				# save into tensorboard
				writer.add_summary(summary_str, RNN.total_step)
				RNN.total_step = RNN.total_step + 2000
				self.SaveSteps()
				self.SaveModel()
				# save model
				print(j, "loss:", loss)
				saver.save(sess=self.sess, save_path="../Model/model.ckpt")

	def LoadModel(self):
		saver = tf.train.Saver(max_to_keep=1)
		saver.restore(sess=self.sess, save_path="../Model/model.ckpt")

	def Test(self,route):
		print('Testing...')
		data_file = open(route, 'r')
		resultlist=[]
		TotalLines = Data.GetLinesNum(route)
		FinishedLines = 0
		while 1:
			line = data_file.readline()
			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			line=line+'«'
			result=self.TestPwdProb(line)
			resultlist.append(result)
			FinishedLines = FinishedLines + 1
			if FinishedLines % 1000 == 0:
				print(str(100 * FinishedLines / float(TotalLines)) + str("%"))

		x=np.arange(0, 25, 0.1)
		y=[]

		for i in x:
			sum = 0
			for j in resultlist:
				if j <pow(10,i):
					sum+=1
			y.append(sum/len(resultlist))

		plt.plot(x, y)
		plt.title('Guess Result')
		plt.xlabel('Guesses 10^x')
		plt.ylabel('percent guessed')
		plt.show()

	def softmax(self,x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)
	def ProbSoftMax(self,Probability):
		for i in range(RNN.batch_size):
			for j in range(RNN.max_sequence_length):
				Probability[i][j] = self.softmax(Probability[i][j]).tolist()
		return Probability

	def TestPwdProb(self, TestString): # Calculate the probability of a certain Test String
		# if its length is equal to RNN.batch_size, calculate directly
		# if larger, calculate first bach_size and then multiply the other locations
		if TestString.__contains__('«') == False:
			TestString = TestString + '«'

		# preparation
		idx2char = Data.GetCharsSet()
		char2idx = {c: i for i, c in enumerate(idx2char)}
		Overlen = len(TestString) - RNN.batch_size - 1
		MaxTimes = 10 ** 50
		# Initial Calculate
		x_data,y_data = Data.Pwd2Batch(TestString,RNN.max_sequence_length)
		x_idx = [[char2idx[c] for c in x_data[k]] for k in range(RNN.max_sequence_length)]
		result = FirstCharProb.GetProb(TestString[0])
		# take the first batch_size locaitons into the RNN
		Probability = self.sess.run(self.outputs, feed_dict={self.X: x_idx})
		Probability = self.ProbSoftMax(Probability.tolist()) # change into a good format
		# display the predicted characters
		# for i in range(RNN.batch_size):
		# 	for j in range(RNN.max_sequence_length):
		# 		print(idx2char[Probability[i][j].index(max(Probability[i][j]))],end='')
		# 	print("",end=" ")
		# multiply all first locations
		for i in range(RNN.max_sequence_length):
			result = result * Probability[i][i][char2idx[TestString[i + 1]]]
		if Overlen == 0: # only has the size of batch_size
			if result == 0:
				return MaxTimes
			return 1/result
		# over-long part calculation
		NextString = TestString[Overlen:len(TestString)]
		x_data, y_data = Data.Pwd2Batch(NextString, RNN.max_sequence_length)
		x_idx = [[char2idx[c] for c in x_data[k]] for k in range(RNN.max_sequence_length)]

		Probability = self.sess.run(self.outputs, feed_dict={self.X: x_idx})
		Probability = Probability.tolist()
		Probability = self.ProbSoftMax(Probability)

		for i in range(Overlen):
			result = result * Probability[RNN.max_sequence_length - i - 1]\
											[RNN.max_sequence_length - i - 1]\
											[char2idx[NextString[RNN.max_sequence_length- i]]]
		if result == 0:
			return MaxTimes
		return 1/result

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
			if temp[9] != '«':
				self.Predict(deepcopy(temp),depth,OutFile)
			else:
				result = ''.join(temp)
				result = result.replace(" ", "")
				result = result[:result.index('«')]
				if OutFile is None:
					print(result)
				else:
					OutFile.write(str(result) + '\n')
	@staticmethod
	def CountParaNum():
		total_parameters = 0
		for variable in tf.trainable_variables():
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			variable_parameters = 1
			for dim in shape:
				variable_parameters *= dim.value
			total_parameters += variable_parameters
		print('Parament Number : ',total_parameters)