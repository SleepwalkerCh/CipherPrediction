from Data import ReadFileDataWithoutHandle
from LSTM import RNN
rnn = RNN()
rnn.CreateNetwork()
data = ReadFileDataWithoutHandle("z.txt_train.txt")
rnn.Train(data,4000)