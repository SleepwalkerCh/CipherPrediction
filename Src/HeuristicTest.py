#coding=utf-8
import re
class HeuristicTest:
    def __init__(self,pwd):
        self.special = "~!@#$%^&*()_+{}|:<>?[]\;',./=-`"
        self.pwd = pwd
    #get A
    def GetLength(self):
        return len(self.pwd)
    #get U_ch
    def GetUpperLength(self):
        return len([item for item in self.pwd if item.isupper() is True])
    #get L_ch
    def GetLowerLength(self):
        return len([item for item in self.pwd if item.islower() is True])
    #get N_ch
    def GetNumLength(self):
        return len([item for item in self.pwd if item.isdigit() is True])
    #get S_ch
    def GetSpecialLength(self):
        return len([item for item in self.pwd if item in self.special])
    #get Mid
    def GetMid(self):
        return len([item for item in self.pwd[1:-1] if item.isdigit() is True or item in self.special])
    #get all_five
    def GetAllFive(self):
        if('0' not in [self.GetLength(),self.GetUpperLength(),self.GetLowerLength(),self.GetNumLength(),self.GetMid()]):
            return 1
        else:
            return 0
    #get Only Low Letter Length
    def GetLowOnlyLength(self):
        if self.pwd.islower() is True:
            return len(self.pwd)
        else:
            return 0
    #get Only Num Length
    def GetNumOnlyLength(self):
        if self.pwd.isdigit() is True:
            return len(self.pwd)
        else:
            return 0
    #get repeating letter length
    def GetReLetterLength(self):
        pwd_set = set(self.pwd)
        return len(self.pwd) - len(pwd_set)
    #get consistent Upper Letter Length
    def ConsUpperLength(self):
        pattern = r"[A-Z]{2,}"
        res = re.findall(pattern,self.pwd)
        res.sort(key=lambda x: -len(x))
        return len(res[0])
    # get consistent Lower Letter Length
    def ConsLowerLength(self):
        pattern = r"[a-z]{2,}"
        res = re.findall(pattern,self.pwd)
        res = re.findall(pattern, self.pwd)
        res.sort(key=lambda x: -len(x))
        return len(res[0])
# get consistent Number Length
    def ConsNumLength(self):
        pattern = r"\d{2,}"
        res = re.findall(pattern,self.pwd)
        res = re.findall(pattern, self.pwd)
        res.sort(key=lambda x: -len(x))
        return len(res[0])
# get consistent keyboard number
# at least 3
    def ConsKeyLength(self):
        #initial
        init_sequence = ["1234567890-=",r"qwertyuiop[]\\","asdfghjkl;\'","zxcvbnm,./","1qaz","2wsx","3edc","4rfv","5tgb","6yhn",
                         "7ujm","8ik,","9ol.","0p;/","-pl,","0okm","9ijn","8uhb","7ygv","6tfc","5rdx","4esz"]
        data = []
        for item in init_sequence:
            data.append(item)
            #upper
            if(item.upper() not in data):
                data.append(item.upper())
            #reverse
            if(item[::-1] not in data):
                data.append(item[::-1])
        #print(data)
        head = 0
        tail = 2
        count = 0
        while(head < len(self.pwd)-2):
            if tail > len(self.pwd):
                break
            for item in data:
                temp = tail
                while True:
                    if self.pwd[head:tail] in item:
                        if (tail >= len(self.pwd)):
                            break
                        tail += 1
                    else:
                        if tail > temp:
                            tail -= 1
                        break
                if(tail - head > 2):
                    ##print(tail)
                    count += tail - head
                    head = tail-1
                    tail += 2
                    #alreay found
                    break
                else:
                    tail = temp
            head += 1
        return count
    #get digital sequence length
    # at least 3
    def DigitalSeqLength(self):
        pattern = r"\d{3,}"
        res = re.findall(pattern, self.pwd)
        return len(res)
    #get special character length
    def SpecialSeqLength(self):
        data = ["~!#$%^&*()_+/*"]
        head = 0
        tail = 2
        count = 0
        while (head < len(self.pwd) - 2):
            if tail > len(self.pwd):
                break
            for item in data:
                # search all item
                temp = tail
                while True:
                    if self.pwd[head:tail] in item:
                        if (tail >= len(self.pwd)):
                            break
                        tail += 1
                    else:
                        if tail > temp:
                            tail -= 1
                        break
                if (tail - head > 2):
                    ##print(tail)
                    count += tail - head
                    head = tail - 1
                    tail += 2
                    # alreay found
                    break
                else:
                    tail = temp
            head += 1
        return count
    #calculate the score
    def CalculateScore(self):
        A = self.GetLength()
        #print(A)
        U_CH = self.GetUpperLength()
        #print(U_CH)
        L_CH = self.GetLowerLength()
        #print(L_CH)
        N_CH = self.GetNumLength()
        #print(N_CH)
        S_CH = self.GetSpecialLength()
        #print(S_CH)
        Mid = self.GetMid()
        #print(Mid)
        R = self.GetAllFive()
        #print(R)
        OL = self.GetLowOnlyLength()
        #print(OL)
        ON = self.GetNumOnlyLength()
        #print(ON)
        RCS = self.GetReLetterLength()
        #print(RCS)
        CU =self.ConsUpperLength()
        #print(CU)
        CL = self.ConsLowerLength()
        #print(CL)
        CN = self.ConsNumLength()
        #print(CN)
        KS = self.ConsKeyLength()
        #print(KS)
        DS = self.DigitalSeqLength()
        #print(DS)
        SS = self.SpecialSeqLength()
        #print(SS)
        ###testpart###
        score = A * 4 + (A - U_CH) * 2 + (A - L_CH) * 2 + N_CH * 4 + S_CH * 2 \
                + Mid * 2 + R * 2 - OL -ON - RCS - CU * 2 \
                - CL * 2 - CN * 2 - KS * 3 - DS * 3 -SS * 3
        #print(score)
        return score

