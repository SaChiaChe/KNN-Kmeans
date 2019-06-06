import sys
import numpy as np
from utils.ReadData import *
from utils.Calculations import *
import matplotlib.pyplot as plt

KList = [1, 3, 5, 7, 9]
GammaList = [0.001, 0.1, 1, 10, 100]

def KNN(Data, Model, K):
	Predictions = []
	for i in Data:
		Dis, ID = FindKNearest(i, Model, K)
		ID = ID.astype(int)
		Y = Model[:,-1][ID]
		Predict = knborAggregate(Y)
		Predictions.append(Predict)

	return Predictions

def Uniform(Data, Model, Gamma):
	Predictions = []
	for i in Data:
		Weight = CalWeight(i, Model, Gamma)
		Y = Model[:,-1]
		Predict = UniformAggregate(Y, Weight)
		Predictions.append(Predict)

	return Predictions

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python KNN.py TrainData TestData")
		print("Example: python KNN.py Data/Train.dat Data/est.dat")
		exit()

	# Read in data
	TrainDataFile, TestDataFile = sys.argv[1], sys.argv[2]
	TrainData = ReadData(TrainDataFile)
	TestData = ReadData(TestDataFile)

	# Run KNN experiment
	TrackKNNEin = []
	TrackKNNEout = []
	for K in KList:
		print("K:", K)

		Prediction = KNN(TrainData, TrainData, K)
		knborEin = CalError(TrainData[:,-1], Prediction)
		TrackKNNEin.append(knborEin)
		print("k-nbor Ein:", knborEin)

		Prediction = KNN(TestData, TrainData, K)
		knborEout = CalError(TestData[:,-1], Prediction)
		TrackKNNEout.append(knborEout)
		print("k-nbor Eout:", knborEout)

	# Run uniform experiment
	TrackUniformEin = []
	TrackUniformEout = []
	for Gamma in GammaList:
		print("Gamma", Gamma)

		Prediction = Uniform(TrainData, TrainData, Gamma)
		UniformEin = CalError(TrainData[:,-1], Prediction)
		TrackUniformEin.append(UniformEin)
		print("Uniform Ein:", UniformEin)

		Prediction = Uniform(TestData, TrainData, Gamma)
		UniformEout = CalError(TestData[:,-1], Prediction)
		TrackUniformEout.append(UniformEout)
		print("Uniform Eout:", UniformEout)

	# Plot
	# P11: Ein(g_k-nbor) v.s. k
	plt.figure("Problem 11")
	plt.title("$E_{in}(g_{k-nbor})\ v.s.\ k$")
	plt.plot(KList, TrackKNNEin)

	# P12: Eout(g_k-nbor) v.s. k
	plt.figure("Problem 12")
	plt.title("$E_{out}(g_{k-nbor})\ v.s.\ k$")
	plt.plot(KList, TrackKNNEout)

	# P13: Ein(g_uniform) v.s. gamma
	plt.figure("Problem 13")
	plt.title("$E_{in}(g_{uniform})\ v.s.\ gamma$")
	plt.plot(GammaList, TrackUniformEin)

	plt.figure("Problem 13-2")
	plt.title("$E_{in}(g_{uniform})\ v.s.\ log_{10}(gamma)$")
	plt.plot(np.log10(GammaList), TrackUniformEin)

	# P14: Eout(g_uniform) v.s. gamma
	plt.figure("Problem 14")
	plt.title("$E_{out}(g_{uniform})\ v.s.\ gamma$")
	plt.plot(GammaList, TrackUniformEout)

	plt.figure("Problem 14-2")
	plt.title("$E_{out}(g_{uniform})\ v.s.\ log_{10}(gamma)$")
	plt.plot(np.log10(GammaList), TrackUniformEout)

	plt.show()
