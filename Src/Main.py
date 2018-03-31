from Src.FirstCharProb import FirstCharProb
from Src.LSTM import RNN
import gc
from Src.Data import Data
from Src.PILSTM import PILSTM

# rnn = RNN()
# rnn.CreateNetwork()

# for i in range(1,100):
#     DataFileRoute = "F:/TrainData/11/data (" + str(i)  + ")" + ".txt"
#     print(DataFileRoute)
#     Data.init(DataFileRoute)
#     rnn.Train()
#     del Data.batches
#     del Data.DataLines
#     gc.collect()
#     RNN.is_new_train = False

# DataFileRoute = "../Data/data_train.txt"
# Data.init(DataFileRoute)
# RNN.is_new_train = False
# rnn.Train()
# del Data.batches
# del Data.DataLines
# gc.collect()



#test personal info nm
pilstm = PILSTM()
pilstm.is_new_train = False
pilstm.LoadSourceModel()
pilstm.CreateNetwork()

DataFileRoute = "../Data/data_train.txt"
Data.init(DataFileRoute)
pilstm.Train()

FirstCharProb.init()
FirstCharProb.RestoreNum('../Data/FirstCharProb')
FirstCharProb.LearnFormFile("../Data/data_train.txt")
FirstCharProb.TransNum2Prob()
# FirstCharProb.PaintProb()
# rnn.LoadModel()
# rnn.Test('../Data/data_test.txt')
pilstm.LoadModel()
pilstm.Test('../Data/data_test.txt')
