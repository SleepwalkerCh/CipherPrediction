from Data import Data
import pickle
from matplotlib import pyplot as plt
import numpy as np

class FirstCharProb:
	FirstCharDic = {}

	@staticmethod
	def init():
		CharSet = Data.GetCharsSet()
		for char in CharSet:
			FirstCharProb.FirstCharDic[char] = 0
	@staticmethod
	def LearnFormFile(Route):
		file = open(Route,'r')
		CharSet = Data.GetCharsSet()
		TotalLines = Data.GetLinesNum(Route)
		step = int(TotalLines / 30)
		FinishedLines = 0
		# record the number of each char shows up
		while 1:
			line = file.readline()
			if len(line) <= 0:
				break
			if line[0] not in CharSet:
				continue
			FirstCharProb.FirstCharDic[line[0]] += 1
			FinishedLines += 1
			if FinishedLines % step == 0:
				print(100 * FinishedLines / float(TotalLines),"%")
		# change number into probability
		TotalNum = 0
		for char in CharSet:
			TotalNum += FirstCharProb.FirstCharDic[char]
		for char in CharSet:
			FirstCharProb.FirstCharDic[char] /= TotalNum

	@staticmethod
	def SaveProb(Route):
		file = open(Route, 'wb')
		pickle.dump(FirstCharProb.FirstCharDic, file)
		file.close()

	@staticmethod
	def RestoreProb(Route):
		file = open(Route, 'rb')
		FirstCharProb.FirstCharDic = pickle.load(file)
		file.close()
	@staticmethod
	def PaintProb():
		fig = plt.figure(1,figsize=(16, 5))
		ax1 = plt.subplot(111)
		data = []
		CharSet = Data.GetCharsSet()
		for char in CharSet:
			data.append(FirstCharProb.FirstCharDic[char])
		data = np.array(data)
		width = 1
		x_bar = np.arange(len(FirstCharProb.FirstCharDic))
		ax1.bar(left=x_bar, height=data,align="center",alpha=0.5, width=width, color="green")

		ax1.set_xticks(x_bar)
		ax1.set_xticklabels(CharSet)
		ax1.set_ylabel("Probability")
		ax1.set_title("First Char Probability")
		ax1.grid(True)
		ax1.set_ylim(0, FirstCharProb.FirstCharDic[max(FirstCharProb.FirstCharDic, key=FirstCharProb.FirstCharDic.get)])
		plt.show()

	@staticmethod
	def GetProb(char):
		return FirstCharProb.FirstCharDic[char]
