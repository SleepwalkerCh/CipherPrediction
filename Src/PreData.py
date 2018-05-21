#coding=utf-8
import copy
from base64 import test
import itertools
from pypinyin import lazy_pinyin
import random
import re



class PreData:
	def handle_data(self, lastname="", firstname="", birthday="", ID_last4="", qq_number="", mobilenum="", phonenum="",
					carnum="", studentnum="", other=""):
		# print(lazy_pinyin(u'王征'))
		# lastname = "慕容"
		# 姓信息块,输入格式：“慕容”，输出2
		lastname_infos = self.extract_xname(lastname)
		#print(lastname_infos)
		# 名信息块，输入格式：“云海”，输出2
		firstname_infos = self.extract_xname(firstname)
		# print(lastname_infos)
		name_infos = list()
		#print(lastname_infos[0] + firstname_infos[0])
		#print(lastname_infos[0].join(firstname_infos[0]))
		if (lastname != "" and firstname != ""):
			name_infos.append(lastname_infos[0] + firstname_infos[0])
			name_infos.append(lastname_infos[1] + firstname_infos[1])
		# 生日信息块，输入格式：“19980201”，输出6
		# birthday = "19980811"
		birthday_infos = self.extract_bday(birthday)
		# 身份证模块，输出1
		id_infos = [ID_last4]
		# QQ号模块，输出1
		qq_infos = [qq_number]
		# 手机号模块，输入格式：“18811778322”，输出3
		if (mobilenum == ""):
			mobile_infos = [""]
		else:
			mobile_infos = [mobilenum[-4:], mobilenum]
		# 电话模块，输入格式：“0393-8960012”，输出2
		# phonenum = "0393-8960012"
		if (phonenum == ""):
			phone_infos = [""]
		else:
			phone_infos = self.extract_phone(phonenum)
		# print(phone_infos)
		# 车牌模块，输入格式：“JA6931”，输出：“JA, 6931”，输出2
		if (carnum == ""):
			car_infos = [""]
		else:
			alpha = ''.join([item for item in carnum if item.isalpha()])
			dig = ''.join([item for item in carnum if item.isdigit()])
			car_infos = [alpha, dig]
		# 学号或者工号
		# 取后5位数
		if (studentnum == ""):
			student_infos = [""]
		else:
			student_infos = [studentnum[-5:]]
		# 其余信息，输出1
		if (other == ""):
			other_infos = [""]
		else:
			other_infos = [other]
		res = list()
		res.extend(lastname_infos)
		res.extend(firstname_infos)
		res.extend(birthday_infos)
		res.extend(id_infos)
		# print(res)
		res.extend(mobile_infos)
		# print(res)
		res.extend(phone_infos)
		res.extend(car_infos)
		res.extend(student_infos)
		res.extend(other_infos)
		while ('' in res):
			res.remove('')
		#print(res)
		self.info_pieces = res
		file = open('../Data/data_temp.txt', 'w')
		train_file = open('../Data/temp_data.txt', 'w')

		for i in range(1, 5):
			rawdata = list(itertools.permutations(res, i))
			# print(rawdata)
			for item in rawdata[:-1]:
				str = ''.join(item)
				if len(str) > 7 and len(str) < 16:
					train_file.write('\n' + str)
					# judge if str contains both wang and Wang
					str_list = self.JudgeAndSplit(str)
					if (str_list is None):
						continue
					for item in str_list:
						item += '\n'
						file.write(item)
			train_file.write('\n' + ''.join(rawdata[-1]))
			str_list = self.JudgeAndSplit(''.join(rawdata[-1]))
			for item in str_list[:-1]:
				item += '\n'
				file.write(item)
			file.write(''.join(str_list[-1]))
		train_file.close()
		file.close()
		# delete repeated data
		src_file = open('../Data/data_temp.txt', 'r')
		des_file = open('../Data/data.txt', 'w')
		res_set = set()
		for line in src_file.readlines():
			res_set.add(line)
		res_list = list(res_set)
		# print(res_list)
		for item in res_list[:-1]:
			des_file.write(item)
		#print(res_list[-1].replace('\n', ''))
		des_file.write(res_list[-1].replace('\n', ''))
		src_file.close()
		des_file.close()

		res.extend(name_infos)
		while ('' in res):
			res.remove('')
		piece_file = open('../Data/info_pieces.txt', 'w')
		for item in res[:-1]:
			piece_file.write(item + '\n')
		piece_file.write(res[-1])
		return res

	def JudgeAndSplit(self, str):
		# if len(str) > 9 && <16
		# then split into two string
		split_list = list()
		for i in range(0, 6, 2):
			if (self.info_pieces[i] in str and self.info_pieces[i + 1] in str):
				return None
		if (len(str) <= 9):
			# just return it
			split_list.append(str)
		elif (len(str) == 16):
			# split into three sequences:0-index [index,index+8]
			index = random.randint(1, 7)
			split_list.append(str[:8])
			split_list.append(str[index:index + 8])
			split_list.append(str[-8:])
		elif (len(str) > 9 and len(str) < 16):
			split_list.append(str[:8])
			split_list.append(str[-8:])
		return split_list

	def extract_xname(self, name):
		if (name == ""):
			return ""
		pattern = re.compile(r'[^aoeiuv]?h?[iuv]?(ai|ei|ao|ou|er|ang?|eng?|ong|a|o|e|i|u|ng|n)?')
		res = list()
		str = name
		while str != '':
			if pattern.search(str) is not None:
				res.append(pattern.search(str).group())
				if str.split(pattern.search(str).group())[1] is not None:
					str = str.split(pattern.search(str).group())[1]
				else:
					break
		res_list = list()
		# murong
		res_list.append(''.join(item for item in res))
		# MuRong
		res_list.append(''.join(item.capitalize() for item in res))
		# YH
		res_list.append(''.join(item[0].capitalize() for item in res))
		# yh
		res_list.append(''.join(item[0] for item in res))
		# capitalize
		return res_list

	def extract_bday(self, birthday):
		#返回格式：
		#1998 0201 98 21 021 201
		if(birthday == ""):
			return ""
		res = []
		res.append(birthday[:4])
		res.append(birthday[4:])
		res.append(birthday[2:4])
		if(int(birthday[4:6]) < 10):
			if(int(birthday[6:8]) < 10):
				res.append(birthday[5]+birthday[7])
				res.append(birthday[4:6]+birthday[7])
				res.append(birthday[5]+birthday[6:8])
			else:
				#0213 -> 213
				res.append(birthday[5] + birthday[6:8])
		else:
			if (int(birthday[6:8]) < 10):
				#1102 -> 112
				res.append(birthday[4:6] + birthday[7])
			else:
				pass
		return res
	def extract_phone(self,phone_num):
		phones = phone_num.split("-")
		return phones

	def random_split_file(self,file_path,divide_num):
		train_file = open('.'.join(file_path.split(".")[:-1])+"_train.txt","w")
		test_file = open('.'.join(file_path.split(".")[:-1])+"_test.txt","w")
		src_file = open(file_path,"r")
		src_list = list()
		#读取数据
		while 1:
			line = src_file.readline()
			if not line:
				break
			else:
				src_list.append(line)
		choose_num_list = random.sample(range(len(src_list)), divide_num)
		test_list = list()
		for index in choose_num_list:
			test_list.append(src_list[index])
		train_list = [item for item in src_list if item not in test_list]
		train_file.writelines(src_list)
		test_file.writelines(test_list[:-1])
		test_file.write(test_list[-1].replace('\n', ''))
		train_file.close()
		test_file.close()

	def SplitInitPwd(self,pwd):
		piece_file = open('../Data/info_pieces.txt','r')
		src_list = list()
		res_list = list()
		# 读取数据
		while 1:
			line = piece_file.readline().replace('\n','')
			if not line:
				break
			else:
				src_list.append(line)
			src_list.sort(key=lambda x: -len(x))
		head = 0
		tail = 1
		while(head < len(pwd)):
			temp = tail
			for item in src_list:
				if tail > len(pwd):
					break
				#wang
				while pwd[head:tail] in item:
					if tail > len(pwd):
						break
					else:
						tail += 1
			if tail > temp:
				tail -= 1
				if(pwd[head:tail] in src_list):
					#ang
					res_list.append(pwd[head:tail])
					head = tail
					tail += 1
				else:
					head += 1
					tail = head + 1
			else:
				head += 1
				tail = head + 1
		#split all
		boundary = list(set(res_list))
		boundary.sort(key=lambda x: -len(x))
		res = list()
		init_res = [pwd]

		for item in boundary:
			for it in init_res:
				split_items = it.split(item)
				init_res.remove(it)
				init_res.extend(split_items)
		init_res.extend(boundary)
		init_res = list(set(init_res))
		if len(init_res) == 1:
			#随机选取位置切分
			while True:
				first_pos = random.randint(1,len(pwd)-1)
				second_pos = random.randint(1,len(pwd)-1)
				if(first_pos != second_pos):
					break
			max_one = max(first_pos,second_pos)
			min_one = min(first_pos,second_pos)
			res.append(init_res[0][:min_one])
			res.append(init_res[0][min_one:max_one])
			res.append(init_res[0][max_one:])
		else:
			res = init_res
		return res

# pda = PreData()
# pda.handle_data('murong','yunhai','19971203','0014','1002992920','13700808760','0678-7352674','JA6931','2015211650','597')
# #print(pda.JudgeAndSplit('murongMuRong123'))
# #pda.SplitInitPwd("cao")
# pda.random_split_file("./data.txt",1000)

