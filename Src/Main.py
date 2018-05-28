from Src.FirstCharProb import FirstCharProb
from Src.LSTM import RNN
import gc
from Src.Data import Data
from Src.PILSTM import PILSTM
from time import time
from Src.PreData import PreData
from Src.BlackList import BlackList
from Src.HeuristicTest import HeuristicTest
# rnn = RNN()
# rnn.CreateNetwork()
# rnn.LoadModel()
# RNN.CountParaNum()

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
######method 1######
# blacklist = BlackList()
# pwd = '1234564'
# if blacklist.JudgeInList(pwd=pwd) == 1:
#     print("in list")
# else:
#     print("not in list")
# ######method 2######
start = time()
print("开始于" + str(start) + "秒")
ht = HeuristicTest("qwerTTTstItKICKME&987FUCKHAHAHA%^&")
print(ht.CalculateScore())
stop = time()
print("结束于" + str(stop) + "秒")
print("花费时间为：" + str(stop - start) + "秒")
#test personal info nm
# from Src.PreData import PreData
#
# pda = PreData()
# pda.handle_data('murong','yunhai','19971203','0014','1002992920','13700808760','0678-7352674','JA6931','2015211650','597')
# #print(pda.JudgeAndSplit('murongMuRong123'))
# #pda.SplitInitPwd("cao")
# pda.random_split_file("../Data/temp_data.txt",1000)
#
from Src.Recommand import Recommand


# pda = PreData()
# pieces = pda.handle_data('xu','peirong','19961229')

# pilstm = PILSTM()
# pilstm.is_new_train = True
# pilstm.LoadSourceModel()
# pilstm.CreateNetwork()
#
# DataFileRoute = "../Data/data.txt"
# Data.init(DataFileRoute)
# pilstm.Train()
#
# FirstCharProb.init()
# FirstCharProb.RestoreNum('../Data/FirstCharProb')
# FirstCharProb.LearnFormFile("../Data/data.txt")
# FirstCharProb.TransNum2Prob()
# # FirstCharProb.PaintProb()
# # rnn.LoadModel()
# # rnn.Test('../Data/data_test.txt')
#
# pilstm.LoadModel()
# # recommand = Recommand(pda,pieces,'xusajk2370',50,pilstm)
#
# pilstm.Test('../Data/temp_data_test.txt')
