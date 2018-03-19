import pickle

import numpy as np
from matplotlib import pyplot as plt

from Src.Data import Data


class FirstCharProb:
	FirstCharDicNum = {}
	FirstCharDicProb = {}
	@staticmethod
	def init():
		CharSet = Data.GetCharsSet()
		for char in CharSet:
			FirstCharProb.FirstCharDicNum[char] = 0
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
			FirstCharProb.FirstCharDicNum[line[0]] += 1
			FinishedLines += 1
			if FinishedLines % step == 0:
				print(100 * FinishedLines / float(TotalLines),"%")

	@staticmethod
	def TransNum2Prob(): # change number into probability
		CharSet = Data.GetCharsSet()
		TotalNum = 0
		for char in CharSet:
			TotalNum += FirstCharProb.FirstCharDicNum[char]
		for char in CharSet:
			FirstCharProb.FirstCharDicProb[char] = FirstCharProb.FirstCharDicNum[char] \
												   / TotalNum

	@staticmethod
	def SaveNum(Route):
		file = open(Route, 'wb')
		pickle.dump(FirstCharProb.FirstCharDicNum, file)
		file.close()

	@staticmethod
	def RestoreNum(Route):
		file = open(Route, 'rb')
		FirstCharProb.FirstCharDicNum = pickle.load(file)
		file.close()
	@staticmethod
	def PaintProb():
		fig = plt.figure(1,figsize=(16, 5))
		ax1 = plt.subplot(111)
		data = []
		CharSet = Data.GetCharsSet()
		for char in CharSet:
			data.append(FirstCharProb.FirstCharDicProb[char])
		data = np.array(data)
		width = 1
		x_bar = np.arange(len(FirstCharProb.FirstCharDicProb))
		ax1.bar(left=x_bar, height=data,align="center",alpha=0.5, width=width, color="green")

		ax1.set_xticks(x_bar)
		ax1.set_xticklabels(CharSet)
		ax1.set_ylabel("Probability")
		ax1.set_title("First Char Probability")
		ax1.grid(True)
		ax1.set_ylim(0, FirstCharProb.FirstCharDicProb[max(FirstCharProb.FirstCharDicProb, key=FirstCharProb.FirstCharDicProb.get)])
		plt.show()

	@staticmethod
	def GetProb(char):
		return FirstCharProb.FirstCharDicProb[char]
