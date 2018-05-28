from random import randrange
from Src.LSTM import RNN
from Src.FirstCharProb import FirstCharProb
from Src.PreData import PreData
import itertools
def Recommand(predata,InfoFragments,UserInput,OutputNum,rnn): # 输入碎片集，用户输入，输出的总个数
	result = []
	NumOfFragments = 3
	AddedCharSet = "1234567890qwertyuiopasdfghjklzxcvbnm"
	SelectInfoFragments = list(itertools.combinations(InfoFragments, NumOfFragments)) # 选出个人信息片段，3个
	UserInputFragments = predata.SplitInitPwd(UserInput)  # 打碎用户输入
	for k in range(len(SelectInfoFragments)):
		RandLoc = int(randrange(0,len(UserInputFragments))) # 随机找个位置
		SelectUserFragment = UserInputFragments[RandLoc] # 选出用户输入的一个碎片
		for i in range(NumOfFragments): # 遍历每个位置，放入用户的碎片
			temp = ""
			for j in range(NumOfFragments):
				if j == i:
					temp = temp + SelectUserFragment
				else:
					temp = temp + SelectInfoFragments[k][j]
			# 加两个随机字符
			if randrange(0, 2) == 0:
				RandLoc = randrange(0,len(AddedCharSet))
				temp = temp + AddedCharSet[RandLoc]
			if randrange(0, 2) == 0:
				RandLoc = randrange(0, len(AddedCharSet))
				temp = temp + AddedCharSet[RandLoc]
			if randrange(0,2) == 0:
				RandLoc = randrange(0, len(AddedCharSet))
				temp = temp + AddedCharSet[RandLoc]
			# 查重后放入
			if temp not in result:
				if len(temp) < 8 or len(temp) > 16:
					continue
				result.append([temp,rnn.TestPwdProb(temp)])
			if len(result) >= OutputNum:
				return sorted(result, key=lambda password: -password[1])
	return sorted(result, key=lambda password: -password[1])
# rnn = RNN()
# rnn.CreateNetwork()
# rnn.LoadModel()
# FirstCharProb.init()
# FirstCharProb.RestoreNum('../Data/FirstCharProb')
# FirstCharProb.TransNum2Prob()
# predata = PreData()
# pieces = predata.handle_data('xu','peirong','19961229')
# recommand = Recommand(pieces,'xusajk2370',50,rnn)
# print(recommand)