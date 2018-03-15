from Data import ReadFileDataWithoutHandle
from Data import TestCorrectRate
from LSTM import RNN
#from Network import RNN
rnn = RNN()
rnn.CreateNetwork()
data = ReadFileDataWithoutHandle("z.txt_train.txt")
#data = [[[' ','0','1','2','3','4','5','6','7','8']],[['0','1','2','3','4','5','6','7','8','9']]]
rnn.Train(data,4000)
# TestCorrectRate('Prediction.txt','z.txt_test.txt')