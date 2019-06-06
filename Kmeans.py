import sys
import numpy as np
from utils.ReadData import *
from utils.Calculations import *
import matplotlib.pyplot as plt

KList = [2, 4, 6, 8, 10]

def Kmeans(Data, Centers):
	while True:
		PreCenter = np.copy(Centers)

		# Step 1: Reclusterize
		Clusters = {}
		for C in Centers:
			Clusters[C.tobytes()] = []

		for i in Data:
			ClosestCenter = FindClosestCenter(i, Centers)
			Clusters[ClosestCenter.tobytes()].append(i)

		# Step 2: Recalculate the centers
		Centers = []
		for i in Clusters:
			NewCenter = CalNewCenter(Clusters[i])
			Centers.append(NewCenter)
		Centers = np.array(Centers)

		# Check termination
		# print(PreCenter == Centers)
		if (PreCenter == Centers).all():
			break

	return Centers

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Format: python Kmeans.py TrainData")
		print("Example: python Kmeans.py Data/NoLabelTrain.dat")
		exit()

	# Read in data
	TrainDataFile = sys.argv[1]
	TrainData = ReadData(TrainDataFile)

	TrackAverage = []
	TrackVariance = []
	for K in KList:
		print("K:", K)
		# Run Experiment 500 times
		TrackError = []
		for _ in range(500):
			# Random k centers
			InitailCenters = RandomCenter(TrainData, K)

			# Run Kmeans
			Centers = Kmeans(TrainData, InitailCenters)
			
			# Calculate error
			Error = CalError_(TrainData, Centers)

			# Track error
			TrackError.append(Error)

		AverageError = np.mean(TrackError)
		TrackAverage.append(AverageError)
		VarianceError = np.var(TrackError)
		TrackVariance.append(VarianceError)

	# Plot
	# P15: Average Ein v.s. k
	plt.figure("Problem 15")
	plt.title("$Average\ E_{in}\ v.s.\ k$")
	plt.plot(KList, TrackAverage)

	# P16: Variance Ein v.s. k
	plt.figure("Problem 16")
	plt.title("$Variance\ E_{in}\ v.s.\ k$")
	plt.plot(KList, TrackVariance)

	plt.show()