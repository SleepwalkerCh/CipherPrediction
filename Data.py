def GetBatch(X,Y,BatchSize,order): # X = [[10],[10],...[10]] Y = [[10],[10],...[10]]
	XX = []
	YY = []
	for j in range(BatchSize):
		XX.append(X[order * BatchSize + j])
		YY.append(Y[order * BatchSize + j])
	return XX,YY

def ReadFileDataWithoutHandle(route):
	print("Reading File.....", end=" ")
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
	return result

def Pwd2Batch(password,max_len): # input: like "01234567",max_len = 8
	# X = [[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7]]
	# Y = [[1],[1,2],[0,1,2],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,5],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6,7,E]]
	X = []
	Y = []
	for i in range(1,max_len):
		X.append(password[0:i] + space(max_len - i))
		Y.append(password[1:i + 1] + space(max_len - i))
	X.append(password[0:max_len])
	Y.append(password[1:max_len+1])
	return X,Y

def space(num):
	result = ""
	for i in range(num):
		result = result + " "
	return result

def GetCharsSet():
	letters = list(" abcdefghijklmnopqrstuvwxyz")
	numbers = list("0123456789")
	symbols = list("~!@#$%^&*()_+{}|:<>?[]\;',./")
	special = list("E")
	result = letters + numbers + symbols + special
	return result

# calculate the rate that how much the predicted data in the test data
def TestCorrectRate(PredictedRoute,TestRoute):
	TestData = ReadFileDataWithoutHandle(TestRoute)
	print('The length of Testdata is ',len(TestData))
	PredictedData = ReadFileDataWithoutHandle(PredictedRoute)
	print('The length of PredictedData is ',len(PredictedData))
	total = len(TestData)
	hit = 0
	for i in range(len(PredictedData)):
		if PredictedData[i] in TestData:
			hit = hit + 1
	print('Correct:',hit,' Correct rate is ',100*(float)(hit) / (float)(total) % 1.00,'%')

def CreateDataFileStartFrom(InRoute,StartChar,OutRoute):
	AllData = ReadFileDataWithoutHandle(InRoute)
	OutFile = open(OutRoute,'w')
	for i in range(len(AllData)):
		if AllData[i][0] == StartChar:
			OutFile.write(AllData[i] + '\n')
	OutFile.close()

def DivideTrainAndTestFile(SrcRoute,DivideRate):
	step = 1 / (float)(DivideRate)
	AllData = ReadFileDataWithoutHandle(SrcRoute)
	TrainFile = open(SrcRoute + '_train.txt', 'w')
	TestFile = open(SrcRoute + '_test.txt', 'w')
	for i in range(len(AllData)):
		if i % step == 0:
			TrainFile.write(AllData[i] + '\n')
		else:
			TestFile.write(AllData[i] + '\n')
	TrainFile.close()
	TestFile.close()