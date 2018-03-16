from Data import Data
from LSTM import RNN
rnn = RNN()
rnn.CreateNetwork()
data = Data.init("train_data.txt")
rnn.Train(10000)