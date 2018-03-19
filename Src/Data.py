import os
import re
import gc

class Data:
	BatchLocation = 0
	DataLines = []
	batches = []
	batch_size = 8

	@staticmethod
	def GetBatch(): # return a batch like [x_idx,y_idx]
		Data.BatchLocation = (Data.BatchLocation + 1) % len(Data.batches)
		return Data.batches[Data.BatchLocation]

	@staticmethod
	def ReadFileDataWithoutHandle(route):
		print("Reading File and Get Lines.....", end=" ")
		result = []
		data_file = open(route, 'r')
		finished = 0
		while 1:
			line = data_file.readline()

			if len(line) <= 0:
				break
			if len(line) < 8 or len(line) > 16:
				continue
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			for i in range(len(line) - 7):
				des = line[i:i + 9]
				if i == len(line) - 8:
					des = des + '«'
				result.append(des)
			finished += 1
			if finished % 10000 == 0:
				print(finished,  "lines finished" )
		print('OK')
		Data.DataLines = result

	@staticmethod
	def Pwd2Batch(password,max_len): # input: like "01234567",max_len = 8
		# X = [[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7]]
		# Y = [[1],[1,2],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7,E]]
		X = []
		Y = []
		for i in range(1,max_len):
			X.append(password[0:i] + Data.space(max_len - i))
			Y.append(password[1:i + 1] + Data.space(max_len - i))
		X.append(password[0:max_len])
		Y.append(password[1:max_len+1])
		return X,Y

	@staticmethod
	def Lines2Batches():
		print("Lines to Batches...", end=' ')
		idx2char = Data.GetCharsSet()
		char2idx = {c: i for i, c in enumerate(idx2char)}

		for i in range(len(Data.DataLines)):
			x_data, y_data = Data.Pwd2Batch(Data.DataLines[i], Data.batch_size)
			x_idx = [[char2idx[c] for c in x_data[k]] for k in range(Data.batch_size)]
			y_idx = [[char2idx[c] for c in y_data[k]] for k in range(Data.batch_size)]
			Data.batches.append([x_idx,y_idx])
			if i % 10000 == 0:
				print(i,"lines to batches")
				gc.collect()
		print('Ok')

	@staticmethod
	def space(num):
		result = ""
		for i in range(num):
			result = result + " "
		return result

	@staticmethod
	def GetCharsSet():
		letters = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
		numbers = list("0123456789")
		symbols = list("~!@#$%^&*()_+{}|:<>?[]\;',./=-`\"")
		special = list("«")
		result = letters + numbers + symbols + special
		return result

	@staticmethod
	def init(TrainDataRoute):
		Data.ReadFileDataWithoutHandle(TrainDataRoute)
		Data.Lines2Batches()

	@staticmethod
	# calculate the rate that how much the predicted Data in the test Data
	def TestCorrectRate(PredictedRoute,TestRoute):
		TestData = Data.ReadFileDataWithoutHandle(TestRoute)
		print('The length of Testdata is ',len(TestData))
		PredictedData = Data.ReadFileDataWithoutHandle(PredictedRoute)
		print('The length of PredictedData is ',len(PredictedData))
		total = len(TestData)
		hit = 0
		for i in range(len(PredictedData)):
			if PredictedData[i] in TestData:
				hit = hit + 1
		print('Correct:',hit,' Correct rate is ',100*(float)(hit) / (float)(total) % 1.00,'%')

	@staticmethod
	def CreateDataFileStartFrom(InRoute,StartChar,OutRoute):
		AllData = Data.ReadFileDataWithoutHandle(InRoute)
		OutFile = open(OutRoute,'w')
		for i in range(len(AllData)):
			if AllData[i][0] == StartChar:
				OutFile.write(AllData[i] + '\n')
		OutFile.close()

	@staticmethod
	def GetLinesNum(Route):
		file = open(Route,'r')
		num = 0
		while 1:
			line = file.readline()
			if len(line) <= 0:
				break
			num = num + 1
		file.close()
		return num
	@staticmethod
	def DivideTrainAndTestFile(SrcRoute,DivideRate):
		# DivideRate refers to how much the train file takes up the whole Data file
		# e.g ('Data.txt',0.2) means 20% train file and 80% test file
		CharsSet = Data.GetCharsSet()
		step = 1 / (float)(DivideRate)
		AllData = []
		file = open(SrcRoute,'r')
		while 1:
			line = file.readline()
			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			IsContinue = False
			for i in range(len(line)):
				if line[i] not in CharsSet:
					IsContinue = True
			if IsContinue == True:
				continue
			AllData.append(line)
		TrainFile = open(SrcRoute.replace('.txt','') + '_train.txt', 'w')
		TestFile = open(SrcRoute.replace('.txt','') + '_test.txt', 'w')
		for i in range(len(AllData)):
			if i % step == 0:
				TrainFile.write(AllData[i] + '\n')
			else:
				TestFile.write(AllData[i] + '\n')
		file.close()
		TrainFile.close()
		TestFile.close()
	@staticmethod
	def PartOfDataFileByLength(SrcRoute,MinLen=8,MaxLen=16):
		# Take some of Data items whose length is between MinLen and MaxLen
		AllData = []
		file = open(SrcRoute, 'r')
		while 1:
			line = file.readline()
			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			if len(line) < MinLen or len(line) > MaxLen:
				continue
			AllData.append(line)
		file.close()
		ResultFile = open(SrcRoute.replace('.txt', '') + '(' + str(MinLen) + '-' + str(MaxLen) + ')' + '.txt', 'w')
		for i in range(len(AllData)):
			ResultFile.write(AllData[i] + '\n')
		ResultFile.close()

	@staticmethod
	def PartOfDataFileByRate(SrcRoute,StartRate,EndRate):# Take part of the SrcRoute file
		# from StartRate to EndRate e.g. ('Data.txt',0.6,0.7)
		AllData = []
		file = open(SrcRoute, 'r')
		while 1:
			line = file.readline()
			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			AllData.append(line)
		file.close()
		ResultFile = open(SrcRoute.replace('.txt','')+ '(' + str(StartRate) + '-'+ str(EndRate) + ')'+ '.txt','w')
		for i in range(int(StartRate * len(AllData)),int(EndRate * len(AllData))):
			ResultFile.write(AllData[i] + '\n')
		ResultFile.close()

	@staticmethod
	def Upper2Lower(SrcRoute): # Change every char in SrcRoute into lower char in the same file
		AllData = []
		CharsSet = Data.GetCharsSet()
		file = open(SrcRoute, 'r')
		while 1:
			line = file.readline()
			IsContinue = False

			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			#line = line.lower()
			for i in range(len(line)):
				if line[i] not in CharsSet:
					IsContinue = True
			if IsContinue == True:
				continue
			AllData.append(line)
		file.close()
		os.remove(SrcRoute)
		file = open(SrcRoute,'w')
		for i in range(len(AllData)):
			file.write(AllData[i] + '\n')
		file.close()

	@staticmethod
	def CleanData():
		#src dir
		dic_path = "C:/Users/amazing/Desktop/163demo/"
		#destination dir
		store_path = "C:/Users/amazing/Desktop/163demo_result/"
		# get all file name
		files = os.listdir(dic_path)
		# print(files)
		for file in files:
			if not os.path.isdir(file):
				if "clean" not in file:
					file_name = dic_path + file
					data_file = open(file_name)
					# create result file
					new_name = store_path + str(file_name.split("/")[-1].split(".")[0]) + "_clean.txt"
					print(new_name)
					result_file = open(new_name, 'w')

					content = data_file.read()
					pattern = re.compile(r'----(.*)')
					passwords = pattern.findall(content)

					s = []
					char_set = Data.GetCharsSet()
					for password in passwords:
						#password = password.lower()
						if set(password).issubset(set(char_set)):
							# clean data
							s.append(password + '\n')
						# print(s)

					data_file.close()
					result_file.writelines(s)
					result_file.close()

#Data.DivideTrainAndTestFile('../Data/密码弱口令字典(8-16).txt',0.2)
#Data.PartOfDataFileByRate('密码弱口令字典.txt',0.4,0.5)
#Data.PartOfDataFileByLength('../Data/密码弱口令字典.txt',8,16)
#Data.Upper2Lower('密码弱口令字典.txt')