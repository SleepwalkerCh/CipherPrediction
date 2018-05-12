#coding=utf-8
class BlackList:
    def JudgeInList(self,pwd):
        #text = open(r"C:\Users\amazing\Documents\GitHub\CipherPrediction\Data\BlackList.txt",'r')
        text = open("../Data/BlackList.txt",'r')
        while True:
            line = text.readline().replace('\n','')
            print(line)
            if line is None or line == '':
                return 0
            else:
                if line == pwd:
                    return 1

