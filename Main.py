from Data import Data
from LSTM import RNN
rnn = RNN()
rnn.CreateNetwork()
data = Data.init("密码弱口令字典(0.4-0.5)_train.txt")
rnn.Train(1000)