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
import measurements

totalResult = []
allTempCCR = []
allTempJaccardSim = []
allTempNMI = []
totalResultFile = open('LABOMNI_POINTWISELSH_totalResult','a')
totalResultFile.write('(segment,numberHFs,numberRadius,AverageOfCCR, AverageOfJaccardSim, AverageOfNMI)')
for numberHFs in [2,3,5,9,15,25,31,51,71,101,151,201]:
	for numberRadius in [4,5,9,15,25,31,51,71,101,151,201]:
		tempCCR = []
		tempJacardSim = []
		tempNMI = []
		for c in range(1,6):
			#------------------------------------------------------------------------------
			# Prepration of the output file, initialization of LSH object and parameters
			#------------------------------------------------------------------------------
			dimensionNumber = 2 # as for now 2 dimensions for the longitude and latitude
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
			# The Trajectory dataset - cross
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

			clusteringResults = [[], [],[],[],[],[],[],[], [], [],[],[],[],[],[]]
			for i, clustering in enumerate(KMeansResults.labels_):
				clusteringResults[clustering].append(i)

			fileContainer.write('\n')
			fileContainer.write(str(clusteringResults))	
			fileContainer.write('\n')
			fileContainer.write(str([len(i) for i in clusteringResults]))	

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
			[1, 6, 16, 23, 35, 40, 77, 80, 84, 102, 106, 118, 121, 125, 130, 133, 161, 198, 206, 208], 
			[184, 185, 191], 
			[70, 89, 94, 207], 
			[22, 34, 37, 44, 51, 52, 56, 59, 62, 81, 90, 95, 99, 104, 111, 116, 123, 143, 145, 170, 179, 204], 
			[71, 76, 93, 202]]

			maxIntersection = 0
			for cluster in clusteringResults:
				finalCount = 0
				for oracleCluster in LABOMNIDatasetTrueClassification:
					tempcount = len(set(oracleCluster).intersection(set(cluster)))
					if finalCount < tempcount:
						finalCount = tempcount
				maxIntersection += finalCount
					
			print maxIntersection/15.0
			
			fileContainer.write('\n')
			fileContainer.write('The clustering Jaccard Similarity is : '+ str(maxIntersection/15.0))

			print measurements.jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults)
			print measurements.ccr(LABOMNIDatasetTrueClassification, clusteringResults)
			fileContainer.write('\n')
			fileContainer.write('The Correct Clustering Rate is : '+ str(measurements.ccr(LABOMNIDatasetTrueClassification, clusteringResults)))
			fileContainer.write('\n')
			fileContainer.write('The clustering Jaccard Similarity is : '+ str(measurements.jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults)))
			print measurements.NMI(LABOMNIDatasetTrueClassification, clusteringResults)
			fileContainer.write('\nThe NMI is : '+ str(measurements.NMI(LABOMNIDatasetTrueClassification, clusteringResults)))
			fileContainer.write('\n--------------------------------------------------------------------')
			tempCCR.append(measurements.ccr(LABOMNIDatasetTrueClassification, clusteringResults))
			tempJacardSim.append(measurements.jaccardSim(LABOMNIDatasetTrueClassification, clusteringResults))
			tempNMI.append(measurements.NMI(LABOMNIDatasetTrueClassification, clusteringResults))
		fileContainer.write('\n')
		fileContainer.write(str(tempCCR))
		fileContainer.write('\n')
		fileContainer.write(str(tempJacardSim))
		fileContainer.write('\n')
		fileContainer.write(str(tempNMI))
		fileContainer.write('\n')
		totalResult.append(('PointwiseLSH',numberHFs,numberRadius,np.average(tempCCR), np.average(tempJacardSim), np.average(tempNMI)))
		totalResultFile.write('\n')
		totalResultFile.write(str(('PointwiseLSH',numberHFs,numberRadius,np.average(tempCCR), np.average(tempJacardSim), np.average(tempNMI))))
		allTempCCR.append(np.average(tempCCR))
		allTempJaccardSim.append(np.average(tempJacardSim))
		allTempNMI.append(np.average(tempNMI))
totalResultFile.write('\n')
totalResultFile.write(str(sorted(allTempCCR, reverse=True)))
totalResultFile.write('\n')
totalResultFile.write(str(sorted(allTempJaccardSim, reverse=True)))
totalResultFile.write('\n')
totalResultFile.write(str(sorted(allTempNMI, reverse=True)))
totalResultFile.write('\n')
totalResultFile.write(str(totalResult))
print totalResult
