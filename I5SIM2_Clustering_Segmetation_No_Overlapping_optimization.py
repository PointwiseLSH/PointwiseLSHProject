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
import measurements
import sys


totalResult = []
allTempCCR = []
allTempJaccardSim = []
allTempNMI = []
totalResultFile = open('I5SIM2_NoOverlapping_FFFFF_totalResult','a')
totalResultFile.write('(segment,numberHFs,numberRadius,AverageOfCCR, AverageOfJaccardSim, AverageOfNMI)')
segmentingPosition = [h for h in range(4,11) if h%2 == 0]
#segmentingPosition = sys.argv[1:]
for segment in segmentingPosition:
	segment = int(segment)
	for numberHFs in [2,3,15,25,51,101,151,201]:
		for numberRadius in [2,3,15,25,51,101,151,201]:
			tempCCR = []
			tempJacardSim = []
			tempNMI = []
			for c in range(1,3):
				#------------------------------------------------------------------------------
				# Prepration of the output file, initialization of LSH object and parameters
				#------------------------------------------------------------------------------
				dimensionNumber = segment
				# numberHFs = 15
				# numberRadius = 351
				usedDataset = 'CVRR_dataset_trajectory_clustering\i5sim2.mat'
				fileContainer = open('Pointwise LSH Clustering Segmentation No Overlapping # of points('+str(dimensionNumber)+') Experiment ('+usedDataset[35:-4]+') ', 'a')
				fileContainer.write('\n')
				fileContainer.write('Welcome to our experiment : (Segmentation No Overlapping) HFs_'+ str(numberHFs)+'_R_'+str(numberRadius))
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
				# The Trajectory dataset - I5SIM2
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
					startingPoints = [x for x in range(len(trajectory)) if x%(dimensionNumber/2) == 0]
					for j in startingPoints:
						if len(trajectory[j:]) < dimensionNumber/2:
							continue
						involvedPoints = []
						for s in range(dimensionNumber/2):
							involvedPoints.append(trajectory[j+s][0])
							involvedPoints.append(trajectory[j+s][1])
						hash = newLsh.index(involvedPoints, loadF=numberRadius)
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

				bucketingResult = queryDictionary

				fileContainer.write('\n')
				fileContainer.write(str( 'Clustering time : '+ time.asctime( time.localtime(time.time()) )))

				fileContainer.write('\n')
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
						newFeatureSpace[trajectory][i] = 1
				print 'End Constructing new feature space : ', time.asctime( time.localtime(time.time()) )	

				totallG = 0 # this is intended to compute G in the space complexity in your method . G is the avergae number of trajectories in each feature space
				for i, key in enumerate(bucketingResult):
					totallG += len(bucketingResult[key])

					
				print 'totallG : ', totallG
				# print newFeatureSpace[1540]
					

				# print 'KMeans Clustering'
				print 'Start clustering : ', time.asctime( time.localtime(time.time()) )	
				KMeansResults = KMeans(init='k-means++', n_clusters=8, n_init=100).fit(newFeatureSpace)
				print 'End clustering : ', time.asctime( time.localtime(time.time()) )	
				# for i in range(20):
					# print KMeansResults.predict(X[i])

					
				# print [KMeansResults.predict(newFeatureSpace[i]) for i in range(10)]

				# print [(i, KMeansResults.labels_[i]) for i in range(20)]


				clusteringResults = [[], [],[],[],[],[],[],[]]
				for i, clustering in enumerate(KMeansResults.labels_):
					clusteringResults[clustering].append(i)


				fileContainer.write('\n')
				fileContainer.write(str(clusteringResults))	
				fileContainer.write('\n')
				fileContainer.write(str([len(i) for i in clusteringResults]))	
				# print clusteringResults[0]
				# print clusteringResults[1]
					
				# print [len(i) for i in clusteringResults]

				I5SIM2DatasetTrueClassification = [[3, 6, 14, 19, 23, 61, 64, 73, 104, 105, 112, 119, 131,141, 144, 145, 151, 153, 162, 168, 170, 181, 182, 200, 213, 227, 235, 246, 255,256, 282, 286, 291, 300, 318, 320, 322, 324, 326, 331, 342, 344, 348, 354, 361,364, 367, 368, 373, 376, 379, 386, 395, 407, 412, 417, 418, 422, 427, 438, 439,441, 445, 447, 450, 453, 454, 462, 469, 474, 490, 491, 503, 518, 526, 541, 542,543, 558, 590, 593, 612, 616, 645, 650, 670, 673, 678, 681, 689, 692, 697, 701,702, 710, 722, 727, 733, 735, 738, 741, 743, 754, 763, 798, 801, 802, 808, 816,820, 825, 827, 836, 837, 846, 850, 853, 864, 872, 875, 876, 880, 887, 888, 889,900, 905, 906, 907, 919, 922, 929, 931, 937, 971, 972, 973, 985, 996, 1024, 1026, 1028, 1044, 1046, 1049, 1054, 1090, 1106, 1112, 1120, 1123, 1128, 1130, 1134,1142, 1150, 1157, 1160, 1165, 1175, 1183, 1195, 1197, 1207, 1208, 1209, 1221, 1241, 1244, 1250, 1257, 1265, 1275, 1277, 1281, 1283, 1301, 1307, 1319, 1327, 1329, 1339, 1347, 1354, 1411, 1413, 1435, 1443, 1455, 1458, 1479, 1489, 1499, 1513,1515, 1517, 1523, 1546, 1552, 1575],
				[10, 16, 35, 40, 58, 68, 71, 77, 81, 93, 118, 156, 157, 178, 186, 187, 195, 208, 211, 217, 221, 225, 231, 234, 239, 245, 253, 260, 272, 275, 283, 284, 288, 293, 299, 304, 310, 313, 316, 323, 329, 337, 349, 363, 372, 375, 390, 394, 398, 411, 431, 452, 471, 478, 483, 486, 500, 522, 523, 525, 527, 529, 559, 580, 592, 600, 605, 609, 624, 628, 631, 639, 640, 642, 647, 663, 677, 685, 699, 705, 707, 711, 713, 716, 721, 732, 737, 745, 748, 758, 767, 777, 797, 799, 811, 817, 838, 842, 851, 870, 882, 892, 909, 911, 930, 935, 942, 948, 960, 965, 1003, 1004, 1005, 1021, 1025, 1031, 1033, 1037, 1038, 1065, 1070, 1076, 1078, 1082, 1083, 1084, 1088, 1092, 1098, 1104, 1118, 1124, 1133, 1167, 1174, 1176, 1202, 1213, 1223, 1227, 1234, 1245, 1253, 1293, 1300, 1308, 1314,1322, 1326, 1333, 1334, 1340, 1343, 1349, 1356, 1367, 1369, 1376, 1379, 1398, 1402, 1420, 1431, 1432, 1436, 1438, 1446, 1450, 1452, 1453, 1459, 1465, 1466, 1473, 1474, 1475, 1485, 1486, 1487, 1491, 1497, 1502, 1503, 1510, 1520, 1521, 1531,1533, 1551, 1559, 1561, 1567, 1568, 1578, 1580, 1581, 1583, 1585, 1586, 1599],
				[0, 11, 17, 30, 44, 48, 54, 76, 84, 87, 100, 101, 106, 113, 115, 129, 130, 148, 155, 172, 189, 190, 201, 204, 220, 226, 236, 237, 242, 244, 247, 249, 251, 259, 264, 268, 274, 285, 292, 298, 301, 303, 312, 346, 351, 352, 356, 360, 378, 410, 419, 420, 434, 449, 456, 458, 460, 464, 465, 466, 467, 479, 487, 497, 498, 512, 540, 545, 547, 548, 554, 557, 568, 576, 579, 585, 594, 598, 599, 611, 613, 621, 622, 638, 657, 660, 671, 682, 683, 686, 725, 731, 747, 749, 753, 768, 772, 773, 776, 779, 781, 784, 788, 791, 792, 796, 814, 824, 843, 845, 847, 848, 857, 859, 884, 902, 904, 917, 920, 921, 941, 949, 968, 975, 990, 1011, 1015, 1018, 1019, 1022, 1023, 1032, 1040, 1056, 1059, 1067, 1081, 1089, 1097, 1100, 1102, 1113, 1125, 1137, 1138, 1140, 1147, 1154, 1156, 1182, 1187, 1188, 1193, 1200, 1205, 1206,1210, 1216, 1219, 1233, 1240, 1256, 1274, 1290, 1292, 1303, 1311, 1337, 1344, 1353, 1361, 1362, 1365, 1371, 1395, 1396, 1399, 1404, 1410, 1419, 1429, 1430, 1481, 1488, 1490, 1504, 1514, 1522, 1525, 1534, 1555, 1556, 1570, 1572, 1584, 1591,1595, 1596, 1597, 1598],
				[24, 32, 33, 37, 41, 45, 47, 49, 50, 59, 63, 75, 78, 96, 138, 139, 143, 147, 150, 152, 154, 158, 161, 174, 177, 179, 199, 202, 210, 223, 233, 252, 262, 265, 266, 269, 270, 271, 279, 308, 317, 319, 321, 330, 332, 336, 338, 339, 345, 357, 371, 374, 377, 388, 404, 421, 426, 448, 480, 481, 493, 502, 505, 516, 520, 531, 550, 552, 561, 565, 571, 582, 615, 617, 618, 626, 633, 648, 649, 652, 659, 668, 672, 680, 694, 695, 734, 742, 764, 787, 789, 790, 803, 813, 821, 829, 831, 832, 834, 844, 854, 855, 860, 865, 878, 881, 893, 896, 901, 918, 924, 934, 946, 951, 959, 962, 963, 967, 977, 978, 983, 992, 999, 1002, 1009, 1012, 1041, 1043, 1061, 1064, 1068, 1074, 1093, 1095, 1111, 1115, 1116, 1122, 1127, 1135, 1152, 1155, 1159, 1163, 1178, 1184, 1186, 1194, 1203, 1220, 1228, 1229,1231, 1237, 1239, 1242, 1243, 1248, 1251, 1264, 1271, 1272, 1278, 1288, 1289, 1298, 1309, 1316, 1324, 1332, 1345, 1366, 1373, 1374, 1381, 1393, 1397, 1409, 1416, 1424, 1425, 1437, 1442, 1451, 1456, 1460, 1464, 1469, 1476, 1482, 1483, 1498,1501, 1509, 1527, 1541, 1548, 1574, 1576, 1592],
				[5, 29, 31, 36, 38, 42, 43, 46, 53, 55, 57, 62, 65, 80, 90, 98, 110, 114, 128, 135, 136, 149, 166, 175, 176, 214, 218, 229, 240, 250, 273, 277, 278, 280, 289, 295, 302, 307, 311, 314, 340, 347, 359, 366, 369, 389, 391, 393, 399, 405, 414, 416, 429, 442, 444, 455, 470, 472, 477, 484, 501, 509, 514, 515, 524, 544, 546, 563, 570, 581, 583, 604, 610, 620, 623, 632, 634, 635, 636, 637, 644, 655, 664, 665, 674, 679, 684, 688, 690, 691, 693, 696, 698, 704, 712, 715, 717, 723, 730, 750, 756, 759, 761, 770, 778, 786, 800, 806, 812, 818, 823, 841, 867, 877, 879, 890, 891, 898, 916, 925, 928, 933, 950, 956, 957, 961, 981, 994, 997, 1006, 1008, 1027, 1036, 1045, 1055, 1063,1069, 1071, 1077, 1085, 1086, 1094, 1119, 1139, 1143, 1144, 1145, 1161, 1166, 1168, 1171, 1180, 1185, 1192, 1198, 1204, 1224, 1226, 1230, 1238, 1246, 1262, 1268, 1286, 1291, 1294, 1296, 1305, 1312, 1315, 1338, 1342, 1346, 1358, 1359, 1375,1383, 1388, 1390, 1394, 1406, 1414, 1415, 1418, 1433, 1448, 1454, 1470, 1472, 1484, 1492, 1506, 1508, 1511, 1518, 1539, 1557, 1558, 1564, 1593],
				[1, 13, 15, 18, 26, 27, 34, 39, 52, 56, 60, 70, 82, 92, 108, 109, 123, 124, 125, 163, 164, 167, 171, 188, 193, 205, 209, 215, 216, 230, 238, 241, 257, 261, 263, 267, 287, 290, 309, 327, 358, 365, 381, 385, 406, 408, 415, 435, 440, 443, 446, 459, 463, 473, 475, 476, 482, 485, 488, 494, 506, 513, 528, 533, 534, 537, 551, 553, 556, 567, 572, 573, 574, 575, 587, 591, 606, 614, 625, 627, 630, 646, 654, 656, 662, 669, 714, 719, 720, 736, 739, 746, 752, 762, 765, 771, 793, 795, 807, 815, 819, 822, 826, 833, 849, 858, 862, 868, 873, 883, 908, 910, 913, 914, 915, 944, 953, 955, 964, 966, 974, 979, 980, 982, 984, 991, 1000, 1034, 1039, 1042, 1050, 1052, 1058, 1079, 1080, 1091, 1103, 1107, 1114, 1126, 1131, 1141, 1146, 1153, 1162, 1164, 1170, 1177, 1179, 1189, 1191, 1196, 1201, 1212, 1225, 1236, 1252, 1254, 1255,1259, 1260, 1261, 1266, 1269, 1273, 1279, 1280, 1282, 1284, 1306, 1317, 1325, 1330, 1348, 1352, 1357, 1372, 1378, 1382, 1384, 1387, 1392, 1401, 1405, 1422, 1461, 1463, 1477, 1478, 1495, 1500, 1526, 1536, 1537, 1542, 1554, 1566, 1569, 1587,1588],
				[8, 20, 21, 22, 25, 28, 51, 66, 67, 74, 79, 88, 97, 102, 116, 117, 120, 122, 126, 127, 132, 133, 137, 140, 146, 159, 169, 180, 183, 184, 191, 194, 197, 198, 212, 222, 232, 248, 254, 258, 276, 294, 296, 297, 305, 306, 341, 353, 362, 370, 380, 382, 384, 397, 402, 409, 423, 428, 432, 437, 457, 461, 492, 495, 496, 508, 517, 532, 539, 549, 555, 562, 564, 569, 577, 588, 595, 597, 602, 603, 608, 667, 675, 676, 687, 703, 706, 708, 709, 718, 728, 729, 740, 744, 751, 760, 769, 785, 804, 809, 828, 830, 861, 863, 886, 895, 899, 927, 938, 947, 952, 970, 976, 986, 987, 988, 989, 993, 995, 998, 1001, 1020, 1057, 1073, 1087, 1096, 1099, 1101, 1105, 1109, 1117, 1121, 1136, 1148, 1149, 1151, 1173, 1190, 1211, 1214, 1215,1218, 1222, 1258, 1285, 1295, 1297, 1299, 1313, 1321, 1328, 1335, 1336, 1341, 1355, 1360, 1363, 1370, 1377, 1380, 1385, 1386, 1389, 1391, 1400, 1403, 1407, 1417, 1421, 1423, 1426, 1427, 1428, 1434, 1439, 1441, 1444, 1447, 1457, 1462, 1471,1480, 1493, 1494, 1496, 1512, 1516, 1524, 1538, 1543, 1544, 1545, 1553, 1560, 1562, 1565, 1579, 1582, 1589, 1594], 
				[2, 4, 7, 9, 12, 69, 72, 83, 85, 86, 89, 91,94, 95, 99, 103, 107, 111, 121, 134, 142, 160, 165, 173, 185, 192, 196, 203, 206, 207, 219, 224, 228, 243, 281, 315, 325, 328, 333, 334, 335, 343, 350, 355, 383, 387, 392, 396, 400, 401, 403, 413, 424, 425, 430, 433, 436, 451, 468, 489, 499, 504, 507, 510, 511, 519, 521, 530, 535, 536, 538, 560, 566, 578, 584, 586, 589, 596, 601, 607, 619, 629, 641, 643, 651, 653, 658, 661, 666, 700, 724, 726, 755, 757, 766, 774, 775, 780, 782, 783, 794, 805, 810, 835, 839, 840, 852, 856, 866, 869, 871, 874, 885, 894, 897, 903, 912, 923, 926, 932, 936, 939, 940, 943, 945, 954, 958, 969, 1007, 1010, 1013, 1014, 1016, 1017, 1029, 1030, 1035, 1047, 1048, 1051, 1053, 1060, 1062, 1066, 1072, 1075, 1108, 1110, 1129, 1132, 1158, 1169,1172, 1181, 1199, 1217, 1232, 1235, 1247, 1249, 1263, 1267, 1270, 1276, 1287, 1302, 1304, 1310, 1318, 1320, 1323, 1331, 1350, 1351, 1364, 1368, 1408, 1412, 1440, 1445, 1449, 1467, 1468, 1505, 1507, 1519, 1528, 1529, 1530, 1532, 1535, 1540,1547, 1549, 1550, 1563, 1571, 1573, 1577, 1590]]

				maxIntersection = 0
				for cluster in clusteringResults:
					finalCount = 0
					for oracleCluster in I5SIM2DatasetTrueClassification:
						tempcount = len(set(oracleCluster).intersection(set(cluster)))
						if finalCount < tempcount:
							finalCount = tempcount
					maxIntersection += finalCount
						

				print maxIntersection/8.0

				fileContainer.write('\n')
				fileContainer.write('The clustering Jaccard Similarity is : '+ str(maxIntersection/8.0))
				tempCCR.append(measurements.ccr(I5SIM2DatasetTrueClassification, clusteringResults))
				tempJacardSim.append(measurements.jaccardSim(I5SIM2DatasetTrueClassification, clusteringResults))
				tempNMI.append(measurements.NMI(I5SIM2DatasetTrueClassification, clusteringResults))
				print measurements.jaccardSim(I5SIM2DatasetTrueClassification, clusteringResults)
				print measurements.ccr(I5SIM2DatasetTrueClassification, clusteringResults)
				fileContainer.write('\n')
				fileContainer.write('The Correct Clustering Rate is : '+ str(measurements.ccr(I5SIM2DatasetTrueClassification, clusteringResults)))
				fileContainer.write('\n')
				fileContainer.write('The clustering Jaccard Similarity is : '+ str(measurements.jaccardSim(I5SIM2DatasetTrueClassification, clusteringResults)))
				print measurements.NMI(I5SIM2DatasetTrueClassification, clusteringResults)
				fileContainer.write('\nThe NMI is : '+ str(measurements.NMI(I5SIM2DatasetTrueClassification, clusteringResults)))
				fileContainer.write('\n--------------------------------------------------------------------')
			fileContainer.write('\n')
			fileContainer.write(str(tempCCR))
			fileContainer.write('\n')
			fileContainer.write(str(tempJacardSim))
			fileContainer.write('\n')
			fileContainer.write(str(tempNMI))
			fileContainer.write('\n')
			totalResult.append((segment,numberHFs,numberRadius,np.average(tempCCR), np.average(tempJacardSim), np.average(tempNMI)))
			totalResultFile.write('\n')
			totalResultFile.write(str((segment,numberHFs,numberRadius,np.average(tempCCR), np.average(tempJacardSim), np.average(tempNMI))))
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

