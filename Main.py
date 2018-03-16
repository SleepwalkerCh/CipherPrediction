from Data import ReadFileDataWithoutHandle
from Data import TestCorrectRate
from LSTM import RNN
rnn = RNN()
rnn.CreateNetwork()
data = ReadFileDataWithoutHandle("z.txt_train.txt")
rnn.Train(data,4000)
# TestCorrectRate('Prediction.txt','z.txt_test.txt')