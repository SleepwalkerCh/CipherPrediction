from Src.FirstCharProb import FirstCharProb
from Src.LSTM import RNN
import gc
from Src.Data import Data
from Src.PILSTM import PILSTM

#rnn = RNN()
#rnn.CreateNetwork()

# for i in range(1,100):
#     DataFileRoute = "F:/TrainData/11/data (" + str(i)  + ")" + ".txt"
#     print(DataFileRoute)
#     Data.init(DataFileRoute)
    #rnn.Train()
    # del Data.batches
    # del Data.DataLines
    # gc.collect()
    # RNN.is_new_train = False


#test personal info nm
pilstm = PILSTM()
pilstm.LoadSourceModel()
pilstm.CreateNetwork()
#
# DataFileRoute = "../Data/wz.txt"
# Data.init(DataFileRoute)
# pilstm.Train()

FirstCharProb.init()
FirstCharProb.RestoreNum('../Data/FirstCharProb')
FirstCharProb.LearnFormFile("../Data/wz.txt")
FirstCharProb.TransNum2Prob()
# FirstCharProb.PaintProb()
pilstm.LoadModel()
pilstm.Test('../Data/wz_test.txt')
