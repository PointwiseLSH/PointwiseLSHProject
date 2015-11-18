# COPYRIGHT 2015 Mohammed AlMalki
# for the used dataset, please refer to http://cvrr.ucsd.edu/bmorris/datasets/dataset_trajectory_clustering.html
from lshash import LSHash
from sklearn.datasets import fetch_mldata, load_iris, load_digits
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import sets
import cPickle
import gzip
import scipy.io
import time
import timeit
import sets
import datetime as dt
import random


#------------------------------------------------------------------------------
# Prepration of the output file, initialization of LSH object and parameters
#------------------------------------------------------------------------------
dimensionNumber = 2 # as for now 2 dimensions for the longitude and latitude
# numberHFs = 67
# numberRadius = 77
for numberHFs in [1,2,5,6,11,21,31,51,71,101,151,201,501]:
	allCCR = []
	allJacardSim = []
	allNMI = []
	for numberRadius in [1,2,5,6,11,21,31,51,71,101,151,201,501]:
		usedDataset = 'CVRR_dataset_trajectory_clustering\labomni.mat'
		fileContainer = open('Pointwise LSH Clustering Experiment(DifferentParametersValues6) ('+usedDataset[35:-4]+')', 'a')
		fileContainer.write('\n')
		fileContainer.write('Welcome to our experiment : '+str(numberHFs)+'-'+str(numberRadius))
		fileContainer.write(str( '\nSTRATING TIME : '+ time.asctime( time.localtime(time.time()) )))
		print str( '\nSTRATING TIME : '+ time.asctime( time.localtime(time.time()) ))
		fileContainer.write('\n')
		fileContainer.write('The discription needed for each result will be provided accordingly .....')
		fileContainer.write('\n')
		fileContainer.write('The used Dataset is : '+usedDataset)
		fileContainer.write('\n')
		print '\nStarting LSH initialization ...'
		fileContainer.write(str( '\nTime before LSH initialization : '+ time.asctime( time.localtime(time.time()) )))
		newLsh = LSHash(numberHFs, dimensionNumber, num_hashtables = 1)
		fileContainer.write(str( '\nTime after LSH initialization : '+ time.asctime( time.localtime(time.time()) )))
		print '\nStarting loading the trajectory dataset ...'
		fileContainer.write(str( '\nTime before loading the trajectory dataset : '+ time.asctime( time.localtime(time.time()) )))
		#------------------------------------------------------------------------------
		# The Trajectory dataset - LABOMNI
		#------------------------------------------------------------------------------
		mat = scipy.io.loadmat(usedDataset)
		datasetSize = len(mat.values()[0])
		trajectoriesContainer = []
		for i in range(datasetSize):
			trajectoriesContainer.append([(mat.values()[0][i][0][0][j], mat.values()[0][i][0][1][j]) for j in range(len(mat.values()[0][i][0][0]))])
		allPoints = []
		fileContainer.write(str( '\nTime after loading trajectory dataset : '+ time.asctime( time.localtime(time.time()) )))
		#------------------------------------------------------------------------------


		# indexing all trajectories
		print '\nStarting the indexing procedure ...'
		fileContainer.write(str( 'Time before indixing all trajectories points : '+ time.asctime( time.localtime(time.time()) )	))
		queryDictionary = {}
		numberOfPoints = 0
		for i, trajectory in enumerate(trajectoriesContainer):
			for point in trajectory:
				hash = newLsh.index(point, loadF=numberRadius)
				# hash = newLsh.index((point[0]*random.gauss(0.99975,0.00025),point[1]*random.gauss(0.99975,0.00025)), loadF=numberRadius)
				if queryDictionary.has_key(hash):
					# queryDictionary[hash].add(i)
					queryDictionary[hash].append(i)
				else:
					# queryDictionary[hash] = set()
					queryDictionary[hash] = []
					# queryDictionary[hash].add(i)
					queryDictionary[hash].append(i)
				numberOfPoints += 1

		fileContainer.write('\nThe following is the hash table used for querying or clustering ...')
		fileContainer.write(str(queryDictionary))
		fileContainer.write('\n')
		fileContainer.write(str( 'The number of generated buckets is : '+ str(len(queryDictionary.keys()))))
		fileContainer.write('\n')
		fileContainer.write(str( 'Time after indixing all trajectories points : '+ time.asctime( time.localtime(time.time()) )	))
		fileContainer.write('\n')
		fileContainer.write(str( 'The number of point have beenindexed is : '+ str(numberOfPoints)))

		bucketingResult = queryDictionary

		# To return it to sets instead of lists
		for key in bucketingResult.keys():
			bucketingResult[key]=set(bucketingResult[key])
		fileContainer.write('\n')
		fileContainer.write(str( 'Clustering time : '+ time.asctime( time.localtime(time.time()) )))

		# fileContainer.write('\n')
		# print len(bucketingResult.keys())
		fileContainer.write('\n')
		fileContainer.write(str(bucketingResult))
		print 'Start Constructing new feature space : ', time.asctime( time.localtime(time.time()) )	

		NFSdimensions = (datasetSize, len(bucketingResult.keys()))

		newFeatureSpace = np.zeros(NFSdimensions)

		# print len(newFeatureSpace)
		# print len(newFeatureSpace[0])
		for i, key in enumerate(bucketingResult):
			for trajectory in bucketingResult[key]:
				newFeatureSpace[trajectory][i] += 25
		print 'End Constructing new feature space : ', time.asctime( time.localtime(time.time()) )	

		totallG = 0 # this is intended to compute G in the space complexity in your method . G is the avergae number of trajectories in each feature space
		for i, key in enumerate(bucketingResult):
			totallG += len(bucketingResult[key])

			
		print 'totallG : ', totallG
		# print newFeatureSpace[1540]
		fileContainer.write('\n')
		fileContainer.write(str( 'totallG : '+ str(totallG)))

		# print 'KMeans Clustering'
		print 'Start clustering : ', time.asctime( time.localtime(time.time()) )	
		KMeansResults = KMeans(init='k-means++', n_clusters=15, n_init=555).fit(newFeatureSpace)
		print 'End clustering : ', time.asctime( time.localtime(time.time()) )	
		# for i in range(20):
			# print KMeansResults.predict(X[i])

			
		# print [KMeansResults.predict(newFeatureSpace[i]) for i in range(10)]

		# print [(i, KMeansResults.labels_[i]) for i in range(20)]


		clusteringResults = [[], [],[],[],[],[],[],[], [], [],[],[],[],[],[]]
		for i, clustering in enumerate(KMeansResults.labels_):
			clusteringResults[clustering].append(i)


		fileContainer.write('\n')
		fileContainer.write(str(clusteringResults))	
		fileContainer.write('\n')
		fileContainer.write(str([len(i) for i in clusteringResults]))	
		# print clusteringResults[0]
		# print clusteringResults[1]
			
		# print [len(i) for i in clusteringResults]

		LABOMNIDatasetTrueClassification = [[2, 47, 54, 73, 122, 131, 139, 162], 
		[3, 7, 9, 12, 14, 17, 24, 26, 28, 30, 42, 45, 64, 67, 69, 85, 96, 107, 110, 113, 126, 135, 165, 186, 199], 
		[146, 147, 149, 151, 153, 155, 176, 180], 
		[181, 187, 188], 
		[0, 5, 18, 32, 33, 39, 49, 57, 61, 63, 72, 78, 79, 88, 101, 105, 108, 119, 124, 127, 129, 132, 141, 160, 166, 168, 195, 197, 203, 205], 
		[21, 36, 41, 48, 50, 53, 55, 58, 60,74, 75, 82, 83, 87, 92, 98, 103, 115, 117, 120, 136, 137, 138, 140, 142, 144, 157, 158, 159, 169, 171, 172, 178, 183, 196, 201], 
		[4, 8, 10, 11, 13, 15, 19, 20,25, 27, 29, 31, 38, 43, 46, 68, 91, 97, 109, 112, 114, 128, 134, 163, 164, 167,173, 200], 
		[65, 193, 194], 
		[66, 86, 148, 150, 152, 154, 156, 174, 175, 190, 192], 
		[100, 177, 182, 189], 
		[1, 6, 16, 23, 35, 40, 77, 80, 84, 102, 106, 118, 121, 125, 130, 133, 161, 198, 206], 
		[184, 185, 191], 
		[70, 89, 94, 207], 
		[22, 34, 37, 44, 51, 52, 56, 59, 62, 81, 90, 95, 99, 104, 111, 116, 123, 143, 145, 170, 179, 204], 
		[71, 76, 93, 202]]


		# maxIntersection = 0
		# for cluster in clusteringResults:
			# finalCount = 0
			# oracleClusterID = -1
			# for oracleCluster in LABOMNIDatasetTrueClassification:
				# tempcount = len(set(oracleCluster).intersection(set(cluster)))
				# if finalCount < tempcount:
					# finalCount = tempcount
					# oracleClusterID = LABOMNIDatasetTrueClassification.index(oracleCluster)
			# maxIntersection += finalCount
			# print finalCount, oracleClusterID
				

		# print maxIntersection/15.0
		# fileContainer.write('\n')
		# fileContainer.write('The clustering Jaccard Similarity is : '+ str((maxIntersection/208.0)*100))


		# Jaccard Similarity

		def	jaccardSim(groundTruth, predictionResult):
			comparisonList = []
			for oracleCluster in groundTruth:
				tempList = []
				for i, predictedCluster in enumerate(predictionResult):
					tempcount = len(set(oracleCluster).intersection(set(predictedCluster)))
					tempList.append((tempcount, i))
				comparisonList.append(tempList)
				

			# print comparisonList	

			labelAssignment = []	
			jaccard = 0
			visitedClusters = []
			visitedPrClusters = []
			ii = 0
			flag = True
			while(flag):
				ii += 1
				tempcount = 0
				maxOCIndex, maxPCIndex = -1, -1
				for i, comlist in enumerate(comparisonList):
					if i in visitedClusters:
						continue
					sorted(comlist, reverse=True)[0]
					if tempcount <= sorted(comlist, reverse=True)[0][0] and sorted(comlist, reverse=True)[0][1] not in visitedPrClusters:
						tempcount = sorted(comlist, reverse=True)[0][0]
						maxOCIndex = i
						maxPCIndex = sorted(comlist, reverse=True)[0][1]
						tempIndex = comlist.index((tempcount,maxPCIndex))

				if (maxOCIndex,maxPCIndex) == (-1,-1):
					for i, l in enumerate(comparisonList):
						del comparisonList[i][l.index(sorted(l, reverse=True)[0])] 
					ii -= 1
					continue
				jaccard += tempcount/float(len(groundTruth[maxOCIndex]))
				visitedClusters.append(maxOCIndex)
				labelAssignment.append((maxOCIndex,maxPCIndex))
				visitedPrClusters.append(maxPCIndex)
				if ii == 15:
					flag = False
			return (jaccard/15.0)*100


		# Correct Clustering Rate CCR

		def	ccr(groundTruth, predictionResult):
			comparisonList = []
			for oracleCluster in groundTruth:
				tempList = []
				for i, predictedCluster in enumerate(predictionResult):
					tempcount = len(set(oracleCluster).intersection(set(predictedCluster)))
					tempList.append((tempcount, i))
				comparisonList.append(tempList)
				

			# print comparisonList	

			labelAssignment = []	
			maxIntersection = 0
			visitedClusters = []
			visitedPrClusters = []
			ii = 0
			flag = True
			while(flag):
				ii += 1
				tempcount = 0
				maxOCIndex, maxPCIndex = -1, -1
				for i, comlist in enumerate(comparisonList):
					if i in visitedClusters:
						continue
					sorted(comlist, reverse=True)[0]
					if tempcount <= sorted(comlist, reverse=True)[0][0] and sorted(comlist, reverse=True)[0][1] not in visitedPrClusters:
						tempcount = sorted(comlist, reverse=True)[0][0]
						maxOCIndex = i
						maxPCIndex = sorted(comlist, reverse=True)[0][1]
						tempIndex = comlist.index((tempcount,maxPCIndex))

				if (maxOCIndex,maxPCIndex) == (-1,-1):
					for i, l in enumerate(comparisonList):
						del comparisonList[i][l.index(sorted(l, reverse=True)[0])] 
					ii -= 1
					continue
				maxIntersection += tempcount
				visitedClusters.append(maxOCIndex)
				labelAssignment.append((maxOCIndex,maxPCIndex))
				visitedPrClusters.append(maxPCIndex)
				if ii == 15:
					flag = False
			return (maxIntersection/208.0)*100
				

		allCCR.append(ccr(LABOMNIDatasetTrueClassification, clusteringResults))
		print ccr(LABOMNIDatasetTrueClassification, clusteringResults)
		allJacardSim.append(jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults))
		print jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults)

		# print labelAssignment	
		fileContainer.write('\n')
		fileContainer.write('The Correct Clustering Rate is : '+ str(ccr(LABOMNIDatasetTrueClassification, clusteringResults)))
		fileContainer.write('\n')
		fileContainer.write('The clustering Jaccard Similarity is : '+ str(jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults)))
		wholeTrueClasses = np.zeros(datasetSize)
		for i, trueClass in enumerate(LABOMNIDatasetTrueClassification):
			for trajectory in trueClass:
				wholeTrueClasses[trueClass] = i
		# print list(wholeTrueClasses[:20])

		wholePredClasses = np.zeros(datasetSize)
		for i, predClass in enumerate(clusteringResults):
			for trajectory in predClass:
				wholePredClasses[predClass] = i

		# print list(wholePredClasses[:20])

		allNMI.append(normalized_mutual_info_score(wholeTrueClasses, wholePredClasses))
		print normalized_mutual_info_score(wholeTrueClasses, wholePredClasses)
		fileContainer.write('\nThe NMI is : '+ str(normalized_mutual_info_score(wholeTrueClasses, wholePredClasses)))
		fileContainer.write('\n--------------------------------------------------------------------')
		fileContainer.write('\n')
fileContainer.write('The allCCR is : '+ str(allCCR))
fileContainer.write('\n')
fileContainer.write('The allCCR5 is : '+ str(sorted(allCCR, reverse=True)))
fileContainer.write('\n')
fileContainer.write('The allJacardSim is : '+ str(allJacardSim))
fileContainer.write('\n')
fileContainer.write('The allJacardSim5 is : '+ str(sorted(allJacardSim, reverse=True)))
fileContainer.write('\n')
fileContainer.write('The allNMI is : '+ str(allNMI))
fileContainer.write('\n')
fileContainer.write('The allNMI5 is : '+ str(sorted(allNMI, reverse=True)))
