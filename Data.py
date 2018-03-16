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
		while 1:
			line = data_file.readline()
			if len(line) <= 0:
				break
			while line[0] == " ":
				line = line[1:len(line) - 1]
			line = line.replace("\n", "")
			line = line.replace("\r", "")
			for i in range(len(line) - 7):
				des = line[i:i + 9]
				if i == len(line) - 8:
					des = des + 'E'
				result.append(des)
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
		print('Ok')

	@staticmethod
	def space(num):
		result = ""
		for i in range(num):
			result = result + " "
		return result

	@staticmethod
	def GetCharsSet():
		letters = list(" abcdefghijklmnopqrstuvwxyz")
		numbers = list("0123456789")
		symbols = list("~!@#$%^&*()_+{}|:<>?[]\;',./")
		special = list("E")
		result = letters + numbers + symbols + special
		return result

	@staticmethod
	def init(TrainDataRoute):
		Data.ReadFileDataWithoutHandle(TrainDataRoute)
		Data.Lines2Batches()

	@staticmethod
	# calculate the rate that how much the predicted data in the test data
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
	def DivideTrainAndTestFile(SrcRoute,DivideRate):
		step = 1 / (float)(DivideRate)
		AllData = Data.ReadFileDataWithoutHandle(SrcRoute)
		TrainFile = open(SrcRoute + '_train.txt', 'w')
		TestFile = open(SrcRoute + '_test.txt', 'w')
		for i in range(len(AllData)):
			if i % step == 0:
				TrainFile.write(AllData[i] + '\n')
			else:
				TestFile.write(AllData[i] + '\n')
		TrainFile.close()
		TestFile.close()