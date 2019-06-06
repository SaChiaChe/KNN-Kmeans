import numpy as np

def CalDistance(A, B):
	return np.linalg.norm(A - B)

def CalWeight(Sample, Model, Gamma):
	Weight = np.array([])
	for i in Model:
		Dis = CalDistance(Sample[:-1], i[:-1])
		Weight = np.append(Weight, [np.exp(-Gamma * Dis**2)])

	return Weight

def knborAggregate(Y):
	return 1 if sum(Y) >= 0 else -1

def UniformAggregate(Y, Weight):
	return 1 if sum(Y * Weight) >= 0 else -1

def FindKNearest(Sample, Model, K):
	Dis = np.array([[0, 0]])
	for i in range(len(Model)):
		Dis = np.append(Dis, [[CalDistance(Sample[:-1], Model[i][:-1]), i]], axis = 0)

	Dis = Dis[1:,:]
	SortID = np.argsort(Dis, axis = 0)[:,0]
	SortDis = Dis[SortID]
	return SortDis[:K, 0], SortDis[:K, 1]

def CalError(Y, Prediciton):
	return sum(Y != Prediciton) / len(Y)

def RandomCenter(Data, K):
	ID = np.random.choice(len(Data), K, replace = False)
	return Data[ID]

def FindClosestCenter(Sample, Centers):
	MinDis = CalDistance(Sample, Centers[0])
	ClosestCenter = Centers[0]

	for C in Centers[1:]:
		Dis = CalDistance(Sample, C)
		if Dis < MinDis:
			MinDis = Dis
			ClosestCenter = C

	return ClosestCenter

def CalNewCenter(Data):
	return sum(Data) / len(Data)

def CalError_(Data, Centers):
	Error = 0.
	for i in Data:
		ClosestCenter = FindClosestCenter(i, Centers)
		Error += CalDistance(i, ClosestCenter)**2

	return Error / len(Data)