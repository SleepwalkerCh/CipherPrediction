from Src.FirstCharProb import FirstCharProb
from Src.LSTM import RNN

from Src.Data import Data

#Data.CleanData()
rnn = RNN()
rnn.CreateNetwork()
data = Data.init("./Data/密码弱口令字典_train.txt")
rnn.Train(10000)
FirstCharProb.init()
FirstCharProb.LearnFormFile("./Data/密码弱口令字典_train.txt")
rnn.Test('./Data/密码弱口令字典_test.txt')
rnn.LoadModel()