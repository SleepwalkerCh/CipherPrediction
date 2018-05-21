import pickle

import tensorflow as tf
import numpy as np
# from tensorflow.nn.rnn import rnn

from Src.Data import Data
import matplotlib.pyplot as plt
from Src.FirstCharProb import FirstCharProb

class PILSTM:
    lstm_file_path = '../Model/LSTM.pb'
    hidden_size = 50
    batch_size = 8
    max_sequence_length = 8
    is_new_train = True
    tensorboard_route = "F:/TrainData/pilogs"
    total_step = 0
    layer_num = 2

    def LstmCell(self, hidden_size):
        # LSTM cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        # Dropout Layer
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
        return lstm_cell

    def LoadSourceModel(self):
        #加载保存的LSTM文件
        with tf.Graph().as_default():
            self.output_graph_def = tf.GraphDef()
            with open(self.lstm_file_path,"rb") as f:
                self.output_graph_def.ParseFromString(f.read())
        #导入最后一层的计算tensor和输入tensor
        self.last_train_tensor, self.input_tensor = tf.import_graph_def(self.output_graph_def, return_elements=['output:0','X:0'])
        print(self.last_train_tensor)
        print(self.input_tensor)


    def CreateNetwork(self):
        #新的网络输入
        id2char = Data.GetCharsSet()
        num_classes = len(id2char)
        learning_rate = 0.005
        self.bottleneck_input = tf.placeholder(tf.float32, [None, self.hidden_size], name='HiddenOut')
        #标准答案
        self.bottleneck_truth = tf.placeholder(tf.int32, [self.batch_size, self.max_sequence_length])



        # #define a lstm basic cell
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # # Dropout Layer
        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=1.0)
        # # multi-LSTM cell
        # mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.LstmCell(self.hidden_size) for _ in range(self.layer_num)],
        #                                          state_is_tuple=True)
        # init_state = mlstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        # #create network
        # #print(self.bottleneck_input)
        # std_inputs = tf.reshape(self.bottleneck_input,[8, 8, self.hidden_size])
        # x_length = [1, 2, 3, 4, 5, 6, 7, 8]
        # outputs, _state = tf.nn.dynamic_rnn(mlstm_cell, inputs=std_inputs,sequence_length=x_length,
        #                                     initial_state=init_state, dtype=tf.float32, time_major=False)
        # X_for_fc = tf.reshape(outputs, [-1, self.hidden_size], name="output")

        #定义一层全连接层
        #self.bottleneck_input = tf.reshape(self.bottleneck_input,[64,self.hidden_size])
        #print(self.bottleneck_input.get_shape())
        outputs = tf.contrib.layers.fully_connected(self.bottleneck_input, num_classes, activation_fn=None)
        # reshape out for sequence_loss
        self.outputs = tf.reshape(outputs, [self.batch_size, -1, num_classes])
        weights = tf.ones([self.batch_size, self.max_sequence_length])
        #print()
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs, targets=self.bottleneck_truth, weights=weights)
        self.loss = tf.reduce_mean(sequence_loss)
        tf.summary.scalar('loss', self.loss)
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        print("OK")

    def SaveSteps(self):
        file = open('../Model/pisteps', 'wb')
        pickle.dump(self.total_step, file)
        file.close()

    def RestoreSteps(self):
        file = open('../Model/pisteps', 'rb')
        self.total_step = pickle.load(file)
        file.close()

    def run_bottleneck_on_x(self,x_idx):
        bottleneck_x = self.sess.run(self.last_train_tensor,{self.input_tensor:x_idx})
        bottleneck_values = np.squeeze(bottleneck_x)
        return bottleneck_values

    def Train(self):
        print("Training...")
        saver = tf.train.Saver(max_to_keep=1)
        merged_summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.tensorboard_route, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        # Start
        #导入数据
        for j in range(len(Data.batches)):
            x_idx, y_idx = Data.GetBatch()
            #对x_idx做进一步处理
            bottleneck_x = self.run_bottleneck_on_x(x_idx)
            self.sess.run(self.train, feed_dict={self.bottleneck_input: bottleneck_x, self.bottleneck_truth: y_idx})
            if j % 2000 == 0:
                summary_str, loss = self.sess.run([merged_summary_op, self.loss]
                                                  , feed_dict={self.bottleneck_input: bottleneck_x, self.bottleneck_truth: y_idx})
                # save into tensorboard
                writer.add_summary(summary_str, self.total_step)
                self.total_step = self.total_step + 2000
                self.SaveSteps()
                #self.SaveModel()
                # save model
                print(j, "loss:", loss)
                saver.save(sess=self.sess, save_path="../Model/pimodel.ckpt")

    def LoadModel(self):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=self.sess, save_path="../Model/pimodel.ckpt")

    def Test(self, route):
        print('Testing...')
        data_file = open(route, 'r')
        resultlist = []
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
            line = line + '«'
            result = self.TestPwdProb(line)
            resultlist.append(result)
            FinishedLines = FinishedLines + 1
            if FinishedLines % 1000 == 0:
                print(str(100 * FinishedLines / float(TotalLines)) + str("%"))

        x = np.arange(0, 25, 0.1)
        y = []

        for i in x:
            sum = 0
            for j in resultlist:
                if j < pow(10, i):
                    sum += 1
            y.append(sum / len(resultlist))

        plt.plot(x, y)
        plt.title('Guess Result')
        plt.xlabel('Guesses 10^x')
        plt.ylabel('percent guessed')
        plt.show()

    def TestPwdProb(self, TestString):  # Calculate the probability of a certain Test String
        # if its length is equal to RNN.batch_size, calculate directly
        # if larger, calculate first bach_size and then multiply the other locations
        if TestString.__contains__('«') == False:
            TestString = TestString + '«'
        # preparation
        idx2char = Data.GetCharsSet()
        char2idx = {c: i for i, c in enumerate(idx2char)}
        Overlen = len(TestString) - self.batch_size - 1
        MaxTimes = 10 ** 50
        # Initial Calculate
        x_data, y_data = Data.Pwd2Batch(TestString, self.max_sequence_length)
        x_idx = [[char2idx[c] for c in x_data[k]] for k in range(self.max_sequence_length)]
        bottleneck_x = self.run_bottleneck_on_x(x_idx)
        result = FirstCharProb.GetProb(TestString[0])
        # take the first batch_size locaitons into the RNN
        Probability = self.sess.run(self.outputs, feed_dict={self.bottleneck_input: bottleneck_x})
        Probability = self.ProbSoftMax(Probability.tolist())  # change into a good format
        # display the predicted characters
        # for i in range(RNN.batch_size):
        # 	for j in range(RNN.max_sequence_length):
        # 		print(idx2char[Probability[i][j].index(max(Probability[i][j]))],end='')
        # 	print("",end=" ")
        # multiply all first locations
        for i in range(self.max_sequence_length):
            result = result * Probability[i][i][char2idx[TestString[i + 1]]]
        if Overlen == 0:  # only has the size of batch_size
            if result == 0:
                return MaxTimes
            return 1 / result
        # over-long part calculation
        NextString = TestString[Overlen:len(TestString)]
        x_data, y_data = Data.Pwd2Batch(NextString, self.max_sequence_length)
        x_idx = [[char2idx[c] for c in x_data[k]] for k in range(self.max_sequence_length)]
        bottleneck_x = self.run_bottleneck_on_x(x_idx)

        Probability = self.sess.run(self.outputs, feed_dict={self.bottleneck_input: bottleneck_x})
        Probability = Probability.tolist()
        Probability = self.ProbSoftMax(Probability)

        for i in range(Overlen):
            result = result * Probability[self.max_sequence_length - i - 1] \
                [self.max_sequence_length - i - 1] \
                [char2idx[NextString[self.max_sequence_length - i]]]
        if result == 0:
            return MaxTimes
        return 1 / result

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def ProbSoftMax(self, Probability):
        for i in range(self.batch_size):
            for j in range(self.max_sequence_length):
                Probability[i][j] = self.softmax(Probability[i][j]).tolist()
        return Probability

