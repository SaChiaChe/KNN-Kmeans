import sys
import numpy as np
import random

def ReadData(DataFile):
	Data = np.loadtxt(DataFile)
	# X = Data[:,:-1]
	# Y = np.array([int(x) for x in Data[:,-1]])
	return Data

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Format: python ReadData.py DataFile")
		exit(0)

	Data = ReadData(sys.argv[1])