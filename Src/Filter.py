import re
import os

import binascii

base_dic_path = "G:/163/2/"
base_store_path = "G:/163/22/"
def GetCharsSet():
    lowletters = list(" abcdefghijklmnopqrstuvwxyz")
    upletters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    numbers = list("0123456789")
    symbols = list("~!@#$%^&*()_+{}|:<>?[]\;',./=-`")
    special = list("E")
    result = upletters + lowletters + numbers + symbols + special
    return result


def FilterData(file_name):
    data_file = open(file_name, mode='r', encoding="ISO-8859-1")
    # create result file
    new_name = base_store_path + str(file_name.split("/")[-2]) + "/" + str(file_name.split("/")[-1].split(".")[0]) + "_clean.txt"
    #new_name.replace("TrainSourceData", "TrainDesData")
    print ("正在处理：" + new_name)
    #new_name = str(name_parts[:-1]) + '_clean.txt'
    result_file = open(new_name, mode='w', encoding="ISO-8859-1")
    passwords = list()
    char_set = GetCharsSet()
    num = 0
    while 1:

        temp_str = ""
        line = data_file.readline()
        #print(line)
        # print(line.encode('utf-8'))
        if not line:
            break
        #print(str(num))
        else:
            if line.encode('utf-8') == ('\n'.encode('utf-8') or '\n,'.encode('utf-8') or '\n|'.encode('utf-8') or '\n\t'.encode('utf-8') or '\n '.encode('utf-8')):
                #print("fuck me")
                continue
            else:
                #print(line)
                line = line[::-1]
                line = line[1:]
                #num += 1
                #print(num)
                while (line[0] == " " or line[0] == "-" or line[0] == "\t" or line[0] == "," or line[0] == "|") and  len(line) > 0:
                    if len(line) == 1:
                        break
                    else:
                        #print(line)
                        line = line[1:]
                for c in line:
                    # print("{0}".format(c))
                    if c == '-' or c == ' ' or c == '\t' or c == ',' or c == '|':
                        break
                    else:
                        temp_str += c
                if not temp_str.encode('utf-8') == '\n'.encode('utf-8'):
                    password = temp_str[::-1].strip('\n')
                    if set(password).issubset(set(char_set)):
                        passwords.append(password+'\n')
                    #print(passwords)
    data_file.close()
    result_file.writelines(passwords)
    result_file.close()
    # content = data_file.read()
    # pattern = re.compile(r'----(.*)|\\s{1,}(.*)|\t(.*)')
    # passwords = pattern.findall(content)
    # temp_pass = list()
    # for password in passwords:
    #     temp_pass.append("".join(list(password)).strip())
    # passwords = temp_pass

def SplitFile(file_name, line_size):
    data_file = open(file_name, mode='r', encoding="ISO-8859-1")
    line_num = 0
    part_num = 0
    s = list()
    while 1:
        line = data_file.readline()
        line_num += 1
        s.append(line)
        if (not line) or not (line_num % line_size):
            part_num += 1
            if (len(s) > 2):
                new_name = base_store_path  + str(file_name.split("/")[-1].split(".")[0]) + "_part_" + str(part_num) + ".txt"
                #new_name.replace("TrainDesData", "TrainPartData")
                #print(new_name)
                #open new file
                new_file = open(new_name, mode='w', encoding="ISO-8859-1")
                new_file.writelines(s)
                new_file.close()
            s.clear()  # clear list
            if not line:
                break
# print(GetCharsSet())
def TraverseFile(dic_path):
    # get all file name
    files = os.listdir(dic_path)
    #print(files)
    for file in files:
        if not os.path.isdir(file):
            # if "clean" not in file:
            #     print(dic_path + file)
                #FilterData(dic_path + file)
            SplitFile(dic_path + file, 50000)

TraverseFile(base_dic_path)



