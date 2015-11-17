from lshash import LSHash
from sklearn.datasets import fetch_mldata, load_iris, load_digits
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

#------------------------------------------------------------------------------
# Prepration of the output file, initialization of LSH object and parameters
#------------------------------------------------------------------------------
dimensionNumber = 2 # as for now 2 dimensions for the longitude and latitude
numberHFs = 15
numberRadius = 27
usedDataset = 'CVRR_dataset_trajectory_clustering\labomni.mat'
runtime = str(dt.datetime.now().timetuple()[1])+str(dt.datetime.now().timetuple()[2])+str(dt.datetime.now().timetuple()[3])+str(dt.datetime.now().timetuple()[4])
fileContainer = open('Pointwise LSH Classification Experiment ('+usedDataset[35:-4]+') at '+runtime+' HFs_'+ str(numberHFs)+'_R_'+str(numberRadius), 'a')
fileContainer.write('\n')
fileContainer.write('Welcome to our experiment : ')
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
		if queryDictionary.has_key(hash):
			queryDictionary[hash].add(i)
		else:
			queryDictionary[hash] = set()
			queryDictionary[hash].add(i)
		numberOfPoints += 1

fileContainer.write('\nThe following is the hash table used for querying or clustering ...')
fileContainer.write(str(queryDictionary))
fileContainer.write('\n')
fileContainer.write(str( 'The number of generated buckets is : '+ str(len(queryDictionary.keys()))))
fileContainer.write('\n')
fileContainer.write(str( 'Time after indixing all trajectories points : '+ time.asctime( time.localtime(time.time()) )	))
fileContainer.write('\n')
fileContainer.write(str( 'The number of point have beenindexed is : '+ str(numberOfPoints)))


# This part is for querying to test the Pointwise algorithm accuracy
print '\nStarting querying the Pointwise algorithm leave-one-out validation for all trajectories in the dataset ...'
allResults = []
totalTime = 0
fileContainer.write('\n')
fileContainer.write(str('Time at starting leave-one-out validation querying all trajectories: '+ time.asctime( time.localtime(time.time()) )))
fileContainer.write('\n')
thresholdIndices = []
for queryID in range(datasetSize):
	startingTime = dt.datetime.now()
	orderedQueryResults = []
	queryTrajectory = trajectoriesContainer[queryID]
	fileContainer.write('------------------------------------------------------------------------------------------------------')
	fileContainer.write('\n')
	
	fileContainer.write('# of points in this trajectory query # ' + str(queryID) + ' is : ')
	fileContainer.write(str(len(queryTrajectory)))
	fileContainer.write(' and the actual class is '+str(mat.values()[3][queryID][0])+'\n')
	fileContainer.write('------------------------------------------------------------------------------------------------------')
	fileContainer.write('\n')
	fileContainer.write('   NN---TRAJECTORY_ID---# OF SHARED BUCKETS---PREDICTED CLASS')
	fileContainer.write('\n')
	queryHash = []
	flag = True
	pointsCounter = 0
	for point in queryTrajectory:
		queryHash.append(newLsh.index(point, loadF=numberRadius))
		queryResult = []
		for hash in queryHash:
			if queryDictionary.has_key(hash):
				queryResult.append(queryDictionary[hash])
		queryResult = [result for inner_list in queryResult for result in inner_list]
		finalResult = []
		for result in set(queryResult):
			finalResult.append((queryResult.count(result), result))
	totalTime += (dt.datetime.now() - startingTime).total_seconds()
	threshold5, threshold3, threshold1 = True, True, True
	threshold1Index, threshold3Index, threshold5Index = -1, -1, -1
	correctAnswers = 0
	for i, result in enumerate(sorted(finalResult, reverse=True)):
		if queryID == result[1]: # This to exclude the query trajectory itself from the comparison
			continue
		if mat.values()[3][result[1]][0] == mat.values()[3][queryID][0]:
			correctAnswers += 1
		if result[0] == 5 and threshold5: # Because I want the onces before it.
			threshold5Index = i
			threshold5 = False
		elif result[0] == 3 and threshold3:
			threshold3Index = i
			threshold3 = False
		elif result[0] == 1 and threshold1: # I might not need this one as all the retrieved ones are more than one
			threshold1Index = i 
			threshold1 = False
			# break
		if result[0] > 1: # threshold to compute the recall based on
			orderedQueryResults.append(result[1])
		# fileContainer.write(' %3d. the query trajectory ID is : %4d| # of shared buckets is : %3d| and its class is : %2d' % (i, result[1], result[0],mat.values()[3][result[1]][0]))
		fileContainer.write(' %3d%11d%19d%19d' % (i, result[1], result[0],mat.values()[3][result[1]][0]))
		fileContainer.write('\n')
	thresholdIndices.append((threshold5Index, threshold3Index, threshold1Index))
	fileContainer.write('\n')
	fileContainer.write('The # of correctly predicted queries is : '+str(correctAnswers))
	fileContainer.write('\n')
	fileContainer.write('The following list is results that share more than one bucket : ')
	fileContainer.write('\n')
	fileContainer.write(str(orderedQueryResults))
	fileContainer.write('\n')
	allResults.append(orderedQueryResults)
fileContainer.write('The average query tinme for all '+str(datasetSize)+' trajectories is : '+ str(totalTime/datasetSize)+' sec')
fileContainer.write('\n# RUNID 47\nLabomniDatasetApproximationNNResults = ')
fileContainer.write(str(allResults))
fileContainer.write('\ntheIndeces = ')
fileContainer.write(str(thresholdIndices))

LabomniDatasetApproximationNNResults = allResults
theIndeces = thresholdIndices

print 'Finishing Time is : ', time.asctime( time.localtime(time.time()) )

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

temptuples = []
for i, index in enumerate(theIndeces): # this to solve if the retrieved index is one then to get that one then the boundary should be 2.
	templist = []
	for j, t in enumerate(index):
		if t == 1:
			templist.append(2)
		else:
			templist.append(theIndeces[i][j])
	temptuples.append(tuple(templist))
theIndeces = temptuples		

	
retrievedCount = 0
allPrecisions = []
fileContainer.write('\nLSH based on the true classes evaluation . . .')
for precisionAt in range(1,100):#[1, 2, 10]:
	countCorrectAnswers = 0
	for i, result in enumerate(LabomniDatasetApproximationNNResults):
		for j in range(precisionAt):
			if j >= len(result): # THIS TO AVOID ERROR IF THE RETRIEVED LIST SMALLLER THAT THE PRECISIONAT NEEDED
				break
			if result[j] in LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]:
				countCorrectAnswers +=1
	fileContainer.write("\n--------------------------------------------------------------------")
	fileContainer.write('\nthe precisionAt '+ str(precisionAt)+ ' is : '+ str(round(countCorrectAnswers / (precisionAt * float(datasetSize)), 3)))
	allPrecisions.append(countCorrectAnswers / (precisionAt * float(datasetSize)))
fileContainer.write('\nAll precision results : \n'+str(allPrecisions))
	
intersectionCount_threshold1, intersectionCount_threshold3, intersectionCount_threshold5 = 0, 0, 0
precisionCounter_threshold1, precisionCounter_threshold3, precisionCounter_threshold5 = 0, 0, 0
retrievedCount_threshold1, retrievedCount_threshold3, retrievedCount_threshold5 = 0, 0, 0
for i, result in enumerate(LabomniDatasetApproximationNNResults):
	intersectionCount_threshold1 += (len(list(set(result).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1])) # 100 IS THE NUMBER OF THE RELEVANT TRAJECTORIES (this is namely recall for each query)
	# print 'intersectionCount_threshold1', intersectionCount_threshold1
	if theIndeces[i][1] > 0: # if it is -1 means that no retrieved results at all.
		intersectionCount_threshold3 += (len(list(set(result[:theIndeces[i][1]]).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))
		# print 'intersectionCount_threshold3', intersectionCount_threshold3
	if theIndeces[i][0] > 0: # if it is -1 means that no retrieved results at all.
		intersectionCount_threshold5 += (len(list(set(result[1:theIndeces[i][0]]).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))
		# print 'intersectionCount_threshold5', intersectionCount_threshold5
	if len(result) > 0: # No need to checking the length as there a condition while generated the result for threshold to be at most 1
		retrievedCount_threshold1 += len(result)
		# print 'retrievedCount_threshold1', retrievedCount_threshold1
		precisionCounter_threshold1 += (len(list(set(result).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(result))	
		# print 'precisionCounter_threshold1', precisionCounter_threshold1
	if len(result[:theIndeces[i][1]]) > 0 and theIndeces[i][1] > 0:
		retrievedCount_threshold3 += len(result[:theIndeces[i][1]])
		# print 'retrievedCount_threshold3', retrievedCount_threshold3
		precisionCounter_threshold3 += (len(list(set(result[:theIndeces[i][1]]).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(result[:theIndeces[i][1]]))
		# print 'precisionCounter_threshold3', precisionCounter_threshold3
	if len(result[:theIndeces[i][0]]) > 0 and theIndeces[i][0] > 0:
		retrievedCount_threshold5 += len(result[:theIndeces[i][0]])
		# print 'retrievedCount_threshold5', retrievedCount_threshold5
		precisionCounter_threshold5 += (len(list(set(result[:theIndeces[i][0]]).intersection(LABOMNIDatasetTrueClassification[mat.values()[3][i][0]-1]))))/float(len(result[:theIndeces[i][0]]))
		# print 'precisionCounter_threshold5', precisionCounter_threshold5

fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\nFor threshold 1 : ')
fileContainer.write('\n--------------------------------------------------------------------')
# Average of retrieved length
averageLength = retrievedCount_threshold1/float(datasetSize) 
# print 'retrievedCount : ', retrievedCount_threshold3
fileContainer.write('\naverageLength is ' + str(round(averageLength, 1)))
fileContainer.write('\n--------------------------------------------------------------------')
#Precision
precision = precisionCounter_threshold1/float(datasetSize) # float(datasetSize) COMPUTATIONS NEED TO BE DIVIDED BY float(datasetSize) TO GET THE OVERALL PRECIAIONS THE SAME FOR THE REST (WEIGHTED OR AVERAGE PRECISION)
fileContainer.write('\nPrecision is : '+ str(round(precision,3)))
fileContainer.write('\n--------------------------------------------------------------------')
#Recall 
recall = intersectionCount_threshold1/float(datasetSize)
fileContainer.write('\nRecall is : '+ str(round(recall, 3)))
fileContainer.write('\n--------------------------------------------------------------------')
#FMeasure
fileContainer.write('\nF-Measure is : '+ str(round((2*recall*precision)/(precision+recall), 3)))
# print '# of responces is : ', len(LabomniDatasetApproximationNNResults)
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\nFor threshold 3 : ')
fileContainer.write('\n--------------------------------------------------------------------')
# Average of retrieved length
averageLength = retrievedCount_threshold3/float(datasetSize)
# print 'retrievedCount : ', retrievedCount_threshold3
fileContainer.write('\naverageLength is ' + str(round(averageLength, 1)))
fileContainer.write('\n--------------------------------------------------------------------')
#Precision
precision = precisionCounter_threshold3/float(datasetSize)
fileContainer.write('\nPrecision is : '+ str(round(precision,3)))
fileContainer.write('\n--------------------------------------------------------------------')
#Recall 
recall = intersectionCount_threshold3/float(datasetSize)
fileContainer.write('\nRecall is : '+ str(round(recall, 3)))
fileContainer.write('\n--------------------------------------------------------------------')
#FMeasure
fileContainer.write('\nF-Measure is : '+ str(round((2*recall*precision)/(precision+recall), 3)))
# print '# of responces is : ', len(LabomniDatasetApproximationNNResults)
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\nFor threshold 5 : ')
fileContainer.write('\n--------------------------------------------------------------------')
# Average of retrieved length
averageLength = retrievedCount_threshold5/float(datasetSize)
# print 'retrievedCount : ', retrievedCount_threshold5
fileContainer.write('\naverageLength is ' + str(round(averageLength, 1)))
fileContainer.write('\n--------------------------------------------------------------------')
#Precision
precision = precisionCounter_threshold5/float(datasetSize)
fileContainer.write('\nPrecision is : '+ str(round(precision,3)))
fileContainer.write('\n--------------------------------------------------------------------')
#Recall 
recall = intersectionCount_threshold5/float(datasetSize)
fileContainer.write('\nRecall is : '+ str(round(recall, 3)))
fileContainer.write('\n--------------------------------------------------------------------')
#FMeasure
fileContainer.write('\nF-Measure is : '+ str(round((2*recall*precision)/(precision+recall), 3)))
# print '# of responces is : ', len(LabomniDatasetApproximationNNResults)
fileContainer.write('\n--------------------------------------------------------------------')
fileContainer.write('\n--------------------------------------------------------------------')