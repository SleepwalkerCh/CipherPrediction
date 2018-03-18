from Data import Data
from LSTM import RNN
from FirstCharProb import FirstCharProb
rnn = RNN()
rnn.CreateNetwork()
data = Data.init("密码弱口令字典_train.txt")
rnn.Train(10000)
FirstCharProb.init()
FirstCharProb.LearnFormFile("密码弱口令字典_train.txt")
rnn.Test('密码弱口令字典_test.txt')