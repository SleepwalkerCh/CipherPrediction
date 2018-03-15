from copy import deepcopy
def MergeSort(A,p,r):
	q = (int)((p + r) / 2)
	if p < r:
		MergeSort(A,p,q)
		MergeSort(A,q + 1,r)
		Merge(A,p,q,r)
def Merge(A,p,q,r):
	MAXVALUE = 9999999999999
	len1 = q - p + 1 # 前半部分的长度
	len2 = r - q # 后半部分的长度
	left = []
	right = []
	for i in range(p,q + 1): # 复制左边
		left.append(A[i])
	for i in range(q + 1,r + 1): # 复制右边
		right.append(A[i])
	left.append(MAXVALUE)
	right.append(MAXVALUE)
	i = 0
	j = 0
	for k in range(p,r + 1):
		if left[i] >= right[j]:
			A[k] = right[j]
			j = j + 1
		else:
			A[k] = left[i]
			i = i + 1
def FindIndex(Matrix,Order):
	Matrix1 = deepcopy(Matrix)
	MergeSort(Matrix1,0,len(Matrix1) - 1)
	return Matrix.index(Matrix1[len(Matrix1) - Order])
