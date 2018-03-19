from Src.FirstCharProb import FirstCharProb
from Src.LSTM import RNN
import gc
from Src.Data import Data

rnn = RNN()
rnn.CreateNetwork()

for i in range(1,100):
	DataFileRoute = "G:/163/data (" + str(i)  + ")" + ".txt"
	print(DataFileRoute)
	Data.init(DataFileRoute)
	rnn.Train()
	del Data.batches
	del Data.DataLines
	gc.collect()
# FirstCharProb.init()
# FirstCharProb.RestoreNum('../Data/FirstCharProb')
# FirstCharProb.LearnFormFile("../Data/密码弱口令字典.txt")
# FirstCharProb.TransNum2Prob()
# FirstCharProb.PaintProb()
# rnn.LoadModel()
#rnn.Test('../Data/密码弱口令字典(8-16)_test.txt')
