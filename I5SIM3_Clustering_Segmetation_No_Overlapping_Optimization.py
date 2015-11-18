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
totalResultFile = open('I5SIM3_NoOverlapping_FFFFF_totalResult','a')
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
				numberHFs = 45
				numberRadius = 51
				usedDataset = 'CVRR_dataset_trajectory_clustering/i5sim3.mat'
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
				# The Trajectory dataset - I5SIM3
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
				KMeansResults = KMeans(init='k-means++', n_clusters=16, n_init=100).fit(newFeatureSpace)
				print 'End clustering : ', time.asctime( time.localtime(time.time()) )	
				# for i in range(20):
					# print KMeansResults.predict(X[i])

					
				# print [KMeansResults.predict(newFeatureSpace[i]) for i in range(10)]

				# print [(i, KMeansResults.labels_[i]) for i in range(20)]


				clusteringResults = [[], [],[],[],[],[],[],[],[], [],[],[],[],[],[],[]]
				for i, clustering in enumerate(KMeansResults.labels_):
					clusteringResults[clustering].append(i)


				fileContainer.write('\n')
				fileContainer.write(str(clusteringResults))	
				fileContainer.write('\n')
				fileContainer.write(str([len(i) for i in clusteringResults]))	
				# print clusteringResults[0]
				# print clusteringResults[1]
					
				# print [len(i) for i in clusteringResults]

				I5SIM3DatasetTrueClassification = [[1, 22, 52, 67, 84, 88, 106, 124, 138, 156, 167, 172, 204, 228, 240, 245, 256, 283, 313, 322, 337, 355, 367, 375, 380, 382, 405, 421, 422, 449, 451, 452, 464, 468, 469, 519, 520, 539, 566, 596, 612, 627, 628, 642, 656, 683, 718, 780, 808, 817, 830, 831, 833, 835, 852, 853, 854, 870, 876, 878, 927, 948, 952, 958, 968, 972, 976, 1005, 1016, 1024, 1058, 1108, 1122, 1123, 1149,1152, 1190, 1217, 1236, 1243, 1244, 1257, 1260, 1325, 1331, 1346, 1348, 1375, 1382, 1390, 1393, 1416, 1433, 1445, 1451, 1491, 1521, 1557, 1580, 1588],
				[36, 71,76, 80, 85, 155, 157, 171, 182, 211, 215, 224, 237, 238, 239, 292, 298, 311, 315, 329, 342, 361, 370, 384, 403, 415, 416, 419, 425, 437, 483, 485, 488, 497, 522, 528, 545, 561, 569, 571, 572, 574, 621, 645, 653, 664, 674, 699, 712, 732, 734, 740, 755, 773, 774, 802, 828, 841, 846, 872, 896, 903, 906, 940, 991, 1006, 1013, 1042, 1074, 1085, 1100, 1125, 1148, 1170, 1179, 1180, 1203, 1237, 1239, 1252, 1256, 1274, 1279, 1289, 1297, 1307, 1353, 1358, 1383, 1415, 1417, 1431, 1438,1449, 1457, 1506, 1542, 1544, 1547, 1579],
				[2, 8, 18, 32, 45, 59, 64, 79, 86, 96, 103, 110, 140, 166, 174, 199, 201, 202, 214, 235, 242, 247, 250, 262, 273, 278, 279, 286, 290, 326, 344, 345, 350, 363, 364, 424, 426, 427, 447, 470, 493, 501, 523, 526, 548, 563, 586, 599, 601, 602, 605, 632, 643, 661, 662, 677, 694, 702, 706, 751, 768, 815, 832, 871, 892, 930, 932, 934, 973, 984, 994, 1010, 1064,1080, 1115, 1133, 1136, 1166, 1173, 1258, 1298, 1302, 1313, 1319, 1320, 1321, 1326, 1328, 1333, 1341, 1349, 1385, 1403, 1478, 1489, 1512, 1556, 1559, 1565, 1572], 
				[12, 25, 29, 97, 102, 107, 116, 119, 132, 141, 158, 178, 181, 252, 253, 309,332, 356, 359, 373, 374, 393, 402, 420, 433, 438, 448, 459, 472, 480, 482, 491,515, 517, 529, 530, 543, 581, 618, 623, 636, 650, 676, 687, 727, 728, 737, 742,787, 793, 807, 857, 858, 863, 882, 904, 907, 957, 969, 970, 971, 975, 1000, 1007, 1011, 1043, 1050, 1056, 1069, 1112, 1141, 1145, 1147, 1151, 1157, 1163, 1172,1246, 1316, 1335, 1352, 1360, 1376, 1387, 1399, 1412, 1468, 1505, 1509, 1510, 1513, 1514, 1517, 1529, 1567, 1576, 1578, 1581, 1583, 1591],[6, 21, 23, 34, 42, 53, 77, 90, 154, 168, 195, 264, 267, 269, 282, 284, 335, 336, 352, 354, 358, 379,400, 434, 444, 453, 463, 509, 511, 554, 556, 559, 580, 591, 594, 595, 615, 640,707, 714, 749, 756, 763, 764, 765, 792, 821, 843, 855, 860, 875, 894, 897, 920,954, 1002, 1012, 1025, 1028, 1057, 1076, 1083, 1109, 1127, 1130, 1134, 1144, 1153, 1160, 1194, 1195, 1198, 1207, 1209, 1242, 1245, 1262, 1265, 1284, 1373, 1392, 1407, 1429, 1446, 1461, 1470, 1477, 1479, 1485, 1511, 1519, 1520, 1534, 1539,1554, 1560, 1566, 1577, 1590, 1599],
				[14, 50, 55, 73, 94, 100, 109, 112, 144, 149, 183, 200, 207, 234, 249, 251, 258, 275, 297, 301, 305, 323, 346, 357, 386, 404, 409, 436, 440, 458, 496, 504, 525, 538, 570, 583, 608, 634, 635, 637, 665, 668, 673, 679, 682, 688, 689, 693, 711, 716, 752, 760, 783, 794, 819, 823, 836, 847, 881, 891, 893, 943, 946, 961, 974, 988, 995, 1001, 1003, 1040, 1075, 1155, 1171, 1228, 1259, 1264, 1269, 1291, 1304, 1305, 1308, 1332, 1343, 1356, 1357, 1398, 1401, 1427, 1428, 1435, 1482, 1483, 1484, 1496, 1533, 1535, 1555, 1574, 1586,1589],
				[5, 19, 46, 63, 68, 82, 111, 120, 129, 131, 145, 151, 152, 170, 176, 184,203, 208, 216, 220, 227, 246, 263, 302, 308, 331, 334, 351, 376, 378, 387, 396,408, 412, 428, 489, 500, 502, 555, 577, 622, 631, 651, 652, 686, 705, 745, 747,782, 798, 827, 874, 884, 885, 928, 937, 939, 956, 983, 1009, 1018, 1030, 1039,1062, 1063, 1070, 1072, 1073, 1084, 1093, 1103, 1107, 1128, 1158, 1168, 1169, 1224, 1227, 1230, 1272, 1315, 1327, 1336, 1388, 1394, 1395, 1400, 1409, 1432, 1447, 1448, 1452, 1453, 1465, 1476, 1503, 1515, 1523, 1525, 1582], 
				[9, 13, 15, 41, 44, 93, 113, 122, 134, 136, 139, 173, 185, 189, 205, 241, 259, 260, 268, 272, 299, 318, 327, 353, 389, 394, 413, 430, 479, 492, 499, 503, 516, 531, 535, 573, 579, 590, 609, 614, 638, 660, 666, 670, 691, 717, 724, 757, 762, 796, 801, 803, 809, 811, 838, 844, 873, 888, 899, 905, 908, 913, 918, 929, 953, 977, 982, 990, 992, 1020, 1021, 1051, 1094, 1116, 1124, 1164, 1183, 1184, 1199, 1208, 1219, 1226,1303, 1306, 1323, 1334, 1345, 1354, 1364, 1374, 1377, 1402, 1422, 1501, 1531, 1540, 1553, 1585, 1592, 1597],
				[3, 20, 26, 39, 40, 57, 70, 78, 87, 92, 126, 133, 143, 147, 153, 160, 190, 198, 206, 210, 254, 270, 280, 295, 314, 321, 338, 339, 347, 360, 365, 399, 401, 429, 435, 471, 475, 487, 490, 505, 506, 507, 565, 600, 624, 625, 649, 654, 659, 675, 709, 713, 719, 722, 766, 775, 784, 790, 820, 849, 887, 914, 938, 941, 960, 962, 999, 1054, 1060, 1095, 1099, 1106, 1139, 1142, 1154, 1165, 1186, 1189, 1197, 1273, 1276, 1295, 1301, 1309, 1314, 1324, 1366, 1368,1379, 1436, 1440, 1450, 1459, 1473, 1500, 1528, 1532, 1558, 1563, 1568],
				[0, 4,27, 56, 66, 91, 117, 118, 121, 142, 194, 209, 221, 236, 243, 244, 248, 277, 293,317, 333, 348, 371, 481, 510, 514, 532, 551, 568, 575, 585, 604, 620, 629, 644,681, 700, 704, 720, 726, 743, 748, 770, 779, 795, 850, 851, 859, 867, 889, 911,924, 933, 949, 955, 967, 989, 998, 1019, 1052, 1098, 1105, 1117, 1126, 1131, 1132, 1138, 1156, 1162, 1167, 1175, 1176, 1200, 1238, 1268, 1277, 1278, 1283, 1292, 1310, 1330, 1339, 1350, 1372, 1391, 1404, 1406, 1419, 1463, 1466, 1480, 1481,1490, 1498, 1526, 1538, 1549, 1562, 1570, 1598],
				[24, 51, 54, 74, 108, 130, 148,186, 196, 226, 230, 261, 281, 294, 304, 307, 349, 362, 372, 383, 417, 465, 477,537, 546, 553, 560, 578, 592, 593, 611, 613, 616, 619, 630, 658, 669, 685, 692,696, 791, 805, 806, 834, 837, 840, 845, 856, 868, 869, 898, 915, 945, 951, 964,987, 1004, 1008, 1015, 1036, 1041, 1045, 1071, 1081, 1082, 1088, 1111, 1119, 1120, 1135, 1143, 1159, 1174, 1177, 1193, 1202, 1205, 1221, 1248, 1253, 1280, 1281, 1290, 1293, 1311, 1312, 1340, 1378, 1397, 1405, 1474, 1475, 1486, 1487, 1516,1518, 1537, 1552, 1573, 1596],
				[16, 28, 37, 104, 128, 159, 164, 175, 187, 188, 212, 217, 223, 255, 312, 341, 343, 392, 397, 398, 406, 410, 418, 454, 455, 461, 462, 478, 494, 495, 512, 540, 550, 558, 597, 626, 633, 729, 735, 741, 750, 761, 781, 797, 799, 814, 822, 824, 879, 900, 910, 966, 979, 981, 993, 996, 1037, 1044,1046, 1053, 1061, 1065, 1077, 1079, 1096, 1097, 1113, 1121, 1146, 1150, 1182, 1185, 1196, 1210, 1235, 1241, 1254, 1263, 1275, 1285, 1287, 1338, 1355, 1359, 1363, 1380, 1381, 1396, 1437, 1441, 1467, 1493, 1494, 1495, 1522, 1541, 1546, 1551,1564, 1575],
				[11, 17, 31, 48, 75, 89, 95, 98, 115, 123, 125, 135, 137, 161, 179, 191, 219, 222, 257, 266, 276, 291, 300, 310, 320, 324, 411, 473, 476, 498, 518, 534, 557, 576, 582, 587, 639, 646, 648, 663, 701, 710, 723, 731, 744, 759, 772, 776, 812, 813, 816, 818, 839, 861, 864, 866, 883, 886, 916, 921, 922, 965, 980, 985, 997, 1031, 1032, 1038, 1047, 1059, 1114, 1137, 1191, 1201, 1206, 1215, 1218, 1223, 1234, 1249, 1251, 1266, 1288, 1317, 1318, 1370, 1371, 1414, 1420, 1430, 1439, 1444, 1460, 1464, 1507, 1508, 1524, 1536, 1543, 1548],
				[7, 10, 35, 58, 61, 69, 99, 146, 163, 165, 192, 213, 231, 233, 274, 287, 328, 330, 366, 377, 390,395, 445, 446, 450, 456, 460, 508, 536, 541, 547, 549, 564, 567, 598, 606, 617,657, 671, 672, 695, 703, 725, 733, 736, 754, 769, 771, 778, 785, 786, 800, 810,829, 865, 877, 880, 909, 919, 931, 935, 942, 963, 986, 1023, 1026, 1055, 1068,1086, 1089, 1101, 1102, 1104, 1110, 1140, 1181, 1212, 1229, 1267, 1270, 1286, 1322, 1337, 1347, 1361, 1362, 1369, 1384, 1411, 1413, 1423, 1426, 1456, 1469, 1472, 1488, 1499, 1530, 1561, 1595],
				[38, 43, 60, 62, 101, 225, 229, 232, 285, 288,289, 306, 316, 319, 340, 368, 381, 423, 439, 441, 457, 467, 474, 521, 533, 542,544, 552, 562, 588, 589, 603, 610, 641, 647, 698, 708, 721, 730, 746, 753, 767,788, 825, 862, 890, 895, 902, 912, 917, 925, 926, 947, 959, 1017, 1022, 1029, 1048, 1049, 1066, 1078, 1091, 1092, 1129, 1187, 1192, 1204, 1213, 1216, 1222, 1225, 1231, 1232, 1233, 1247, 1250, 1261, 1271, 1294, 1296, 1299, 1344, 1367, 1389,1410, 1421, 1424, 1425, 1434, 1442, 1443, 1454, 1455, 1458, 1471, 1492, 1550, 1584, 1587, 1593],
				[30, 33, 47, 49, 65, 72, 81, 83, 105, 114, 127, 150, 162, 169,177, 180, 193, 197, 218, 265, 271, 296, 303, 325, 369, 385, 388, 391, 407, 414,431, 432, 442, 443, 466, 484, 486, 513, 524, 527, 584, 607, 655, 667, 678, 680,684, 690, 697, 715, 738, 739, 758, 777, 789, 804, 826, 842, 848, 901, 923, 936,944, 950, 978, 1014, 1027, 1033, 1034, 1035, 1067, 1087, 1090, 1118, 1161, 1178,1188, 1211, 1214, 1220, 1240, 1255, 1282, 1300, 1329, 1342, 1351, 1365, 1386, 1408, 1418, 1462, 1497, 1502, 1504, 1527, 1545, 1569, 1571, 1594]]

				maxIntersection = 0
				for cluster in clusteringResults:
					finalCount = 0
					for oracleCluster in I5SIM3DatasetTrueClassification:
						tempcount = len(set(oracleCluster).intersection(set(cluster)))
						if finalCount < tempcount:
							finalCount = tempcount
					maxIntersection += finalCount
						

				print maxIntersection/16.0

				fileContainer.write('\n')
				fileContainer.write('The clustering Jaccard Similarity is : '+ str(maxIntersection/16.0))
				tempCCR.append(measurements.ccr(I5SIM3DatasetTrueClassification, clusteringResults))
				tempJacardSim.append(measurements.jaccardSim(I5SIM3DatasetTrueClassification, clusteringResults))
				tempNMI.append(measurements.NMI(I5SIM3DatasetTrueClassification, clusteringResults))
				print measurements.jaccardSim(I5SIM3DatasetTrueClassification, clusteringResults)
				print measurements.ccr(I5SIM3DatasetTrueClassification, clusteringResults)
				fileContainer.write('\n')
				fileContainer.write('The Correct Clustering Rate is : '+ str(measurements.ccr(I5SIM3DatasetTrueClassification, clusteringResults)))
				fileContainer.write('\n')
				fileContainer.write('The clustering Jaccard Similarity is : '+ str(measurements.jaccardSim(I5SIM3DatasetTrueClassification, clusteringResults)))
				print measurements.NMI(I5SIM3DatasetTrueClassification, clusteringResults)
				fileContainer.write('\nThe NMI is : '+ str(measurements.NMI(I5SIM3DatasetTrueClassification, clusteringResults)))
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