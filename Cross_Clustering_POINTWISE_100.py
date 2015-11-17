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


allCCR = []
allJaccardSim = []
allNMI = []
for o in range(100):
	#------------------------------------------------------------------------------
	# Prepration of the output file, initialization of LSH object and parameters
	#------------------------------------------------------------------------------
	dimensionNumber = 2 # as for now 2 dimensions for the longitude and latitude
	numberHFs = 25
	numberRadius = 251
	usedDataset = 'CVRR_dataset_trajectory_clustering/cross.mat'
	fileContainer = open('Pointwise LSH Clustering Experiment -ROUND-100- ('+usedDataset[35:-4]+') HFs_'+ str(numberHFs)+'_R_'+str(numberRadius), 'a')
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

	bucketingResult = queryDictionary

	fileContainer.write('\n')
	fileContainer.write(str( 'Clustering time : '+ time.asctime( time.localtime(time.time()) )))

	fileContainer.write('\n')
	print len(bucketingResult.keys())
	fileContainer.write('\n')
	fileContainer.write(str(bucketingResult))
	print 'Start Constructing new feature space : ', time.asctime( time.localtime(time.time()) )	

	NFSdimensions = (1900, len(bucketingResult.keys()))

	newFeatureSpace = np.zeros(NFSdimensions)

	print len(newFeatureSpace)
	print len(newFeatureSpace[0])
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
	KMeansResults = KMeans(init='k-means++', n_clusters=19, n_init=100).fit(newFeatureSpace)
	print 'End clustering : ', time.asctime( time.localtime(time.time()) )	
	# for i in range(20):
		# print KMeansResults.predict(X[i])

		
	print [KMeansResults.predict(newFeatureSpace[i]) for i in range(10)]

	print [(i, KMeansResults.labels_[i]) for i in range(20)]


	clusteringResults = [[], [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	for i, clustering in enumerate(KMeansResults.labels_):
		clusteringResults[clustering].append(i)


	fileContainer.write('\n')
	fileContainer.write(str(clusteringResults))	
	fileContainer.write('\n')
	fileContainer.write(str([len(i) for i in clusteringResults]))	
	print clusteringResults[0]
	print clusteringResults[1]
		
	print [len(i) for i in clusteringResults]

	crossDatasetTrueClassification = [[36, 45, 48, 95, 122, 127, 213, 227, 252, 258, 281, 289, 296, 301, 327, 332, 360, 373, 466, 503, 511, 550, 561, 564, 582, 605, 615, 626, 690, 733, 736, 754, 766, 789, 794, 801, 810, 874, 888, 893, 906, 918, 937, 938, 973, 1028, 1039, 1050, 1071, 1095, 1097, 1111, 1174, 1199, 1241, 1256, 1270, 1303, 1327, 1335, 1347, 1351, 1374, 1384, 1391, 1396, 1406, 1428, 1429, 1432, 1465, 1470, 1480, 1498, 1502, 1506, 1516, 1565, 1585, 1591, 1607, 1639, 1715, 1716,1718, 1724, 1761, 1781, 1797, 1814, 1816, 1831, 1843, 1846, 1848, 1857, 1859, 1860, 1869, 1892], 
	[9, 12, 67, 84, 100, 110, 132, 137, 151, 159, 165, 205, 206, 235, 254, 265, 267, 279, 349, 362, 363, 376, 388, 396, 410, 422, 431, 474, 490, 566, 590, 618, 643, 646, 657, 663, 685, 688, 707, 711, 724, 725, 740, 755, 774, 783, 822, 829, 832, 838, 855, 868, 884, 892, 909, 922, 951, 970, 977, 983, 1014, 1051, 1057, 1090, 1104, 1158, 1185, 1210, 1252, 1265, 1302, 1320, 1322, 1354, 1363, 1378, 1383, 1413, 1443, 1473, 1478, 1482, 1494, 1549, 1588, 1613, 1629, 1654, 1694, 1695, 1741, 1753, 1755, 1769, 1770, 1776, 1828, 1829, 1844, 1898], 
	[51, 98, 102, 107, 111, 116, 123, 124, 152, 169, 176, 180, 183, 219, 277, 298, 316, 365, 381, 382, 395, 397, 443, 462, 470, 478, 493, 501, 504, 525, 531, 569, 644, 654, 678, 679, 694, 720, 737, 750, 779, 797, 812, 817, 843, 856, 865, 866, 870, 878, 891, 899, 911, 942, 962, 967, 982, 992, 994, 995, 999, 1017, 1022, 1066, 1129, 1133, 1152, 1167, 1184, 1220, 1223, 1236, 1240, 1306, 1331, 1387, 1410, 1449,1459, 1464, 1481, 1483, 1518, 1552, 1559, 1624, 1655, 1709, 1726, 1740, 1767, 1782, 1813, 1856, 1861, 1862, 1865, 1883, 1885, 1897], 
	[7, 30, 32, 37, 71, 99, 120, 128, 149, 171, 188, 189, 198, 201, 208, 214, 287, 325, 342, 378, 405, 406, 444, 498, 502, 526, 527, 535, 554, 576, 584, 594, 607, 631, 656, 665, 670, 722, 727, 756, 773, 798, 811, 849, 876, 882, 924, 941, 1026, 1030, 1038, 1061, 1082, 1094, 1100, 1105, 1115, 1179, 1181, 1198, 1200, 1215, 1221, 1228, 1237, 1242, 1254, 1276, 1297, 1324, 1342, 1381, 1394, 1457, 1509, 1514, 1572, 1596, 1625, 1628, 1630, 1645, 1666, 1669, 1673, 1683, 1688, 1697, 1699, 1766, 1771, 1783, 1793, 1805, 1820, 1855, 1879, 1884, 1890, 1896], 
	[6, 23, 46, 69, 104, 112, 121, 143, 187, 221, 229, 243, 255, 308, 313, 317, 324, 339, 345, 371, 404, 409, 412, 452, 492, 506, 529, 616, 625, 637, 661, 667, 684, 703, 708, 730, 731, 758, 769, 775, 806, 821, 828, 872, 894, 926, 936, 1020, 1021, 1035, 1040, 1058, 1072, 1078, 1086, 1089, 1107, 1110, 1112, 1117, 1145, 1193, 1203, 1225, 1247, 1346, 1375, 1386, 1419, 1424, 1434, 1442, 1444, 1456, 1475, 1491, 1515, 1533, 1543, 1551, 1577, 1578, 1601, 1611, 1632, 1633, 1635, 1649, 1651, 1668, 1686, 1696, 1748, 1750, 1752, 1784, 1794, 1840, 1858, 1877], 
	[8, 11, 22, 33, 34, 41, 76, 85, 91, 118, 167, 197, 248, 270, 283, 304, 355, 358, 429, 432, 434, 438, 446, 491, 509, 514, 523, 524, 534, 560, 567, 592, 600, 622, 632, 634, 672, 693, 698, 744, 746, 764, 765, 827, 831, 847, 867, 877, 881, 886, 890, 900, 905, 917, 944, 969, 979, 1010, 1081, 1092, 1116, 1120, 1135, 1160, 1163, 1169, 1170, 1204, 1224, 1234, 1263, 1272, 1278, 1287, 1300, 1312, 1314, 1334, 1339, 1361, 1405, 1487, 1495, 1504, 1547, 1548,1564, 1568, 1583, 1631, 1677, 1725, 1730, 1800, 1809, 1832, 1841, 1847, 1880, 1887], 
	[15, 49, 62, 87, 113, 119, 146, 162, 184, 191, 196, 199, 241, 257, 259, 269, 271, 286, 307, 311, 336, 340, 348, 354, 390, 415, 417, 419, 471, 528, 541, 545, 552, 568, 573, 599, 613, 633, 641, 658, 706, 714, 761, 781, 820, 837, 839, 850, 873, 932, 946, 953, 974, 978, 984, 1011, 1012, 1024, 1027, 1049, 1055, 1056, 1079, 1109, 1125, 1156, 1187, 1207, 1250, 1282, 1305, 1316, 1323, 1366, 1389, 1461, 1474, 1492, 1503, 1505, 1519, 1524, 1527, 1550, 1553, 1569, 1582, 1600, 1602, 1603, 1616, 1619, 1657, 1684, 1689, 1702, 1743, 1754, 1803, 1818], 
	[44, 57, 86, 92, 94, 103, 106, 126, 129, 166, 170, 192, 207, 217, 224, 263, 272, 284, 288, 300, 334, 357, 367, 398, 402, 403, 448, 449, 472, 482, 485, 487, 518, 537, 563, 578, 604, 608, 624, 662, 695, 704, 717, 723, 743, 762, 795, 802, 819, 823, 852, 853, 901, 902, 910, 912, 923, 928, 939, 963, 997, 1008, 1052, 1074, 1114, 1126, 1128, 1132, 1142, 1176, 1178, 1202, 1264, 1266, 1288, 1358, 1360, 1392, 1409, 1415, 1416, 1417, 1420, 1422, 1500, 1538, 1539, 1584, 1586, 1610, 1638, 1661, 1671, 1723, 1733, 1790, 1823, 1839, 1867, 1868], 
	[26, 47, 59, 73, 74, 93, 101, 131, 136, 142, 147, 150, 190, 211, 212, 238, 249, 280, 305, 377, 387, 420, 430, 442, 445, 450, 479, 486, 500, 532, 543, 547, 549, 571, 586, 603, 611, 620, 628, 636, 638, 639, 664, 673, 682, 696, 745, 768, 799, 800, 804, 857, 879, 913, 927, 961, 996, 1007, 1013, 1034, 1062, 1064, 1080, 1087, 1143, 1151, 1159, 1196, 1231, 1239, 1326, 1341, 1349, 1356, 1379, 1430, 1476, 1485, 1525, 1562, 1563, 1575, 1620,1623, 1647, 1691, 1701, 1706, 1708, 1739, 1773, 1779, 1799, 1806, 1842, 1849, 1864, 1875, 1881, 1891], 
	[5, 10, 21, 68, 80, 90, 108, 115, 168, 179, 193, 223, 233, 245, 250, 303, 314, 315, 320, 347, 352, 372, 375, 416, 447, 451, 483, 499, 515, 548, 556, 562, 565, 593, 677, 700, 715, 718, 738, 787, 814, 815, 816, 824, 830, 863, 955, 985, 1016, 1029, 1046, 1065, 1069, 1070, 1077, 1083, 1131, 1147, 1206, 1216, 1227, 1249, 1258, 1271, 1307, 1318, 1333, 1359, 1364, 1421, 1490, 1507, 1521, 1532, 1536, 1573, 1580, 1589, 1609, 1612, 1615, 1634, 1637, 1656, 1659, 1664, 1667, 1674, 1690, 1704, 1717, 1727, 1751, 1757, 1775, 1785, 1819, 1852, 1870, 1889], 
	[13, 54, 61, 63, 72, 139, 178, 232, 247, 262, 293, 306, 310, 322, 331, 333, 335, 380, 414, 421, 439, 473, 494, 516, 546, 551, 553, 581, 589, 606, 619, 675, 697, 728, 747, 763, 770, 777, 864, 875, 880, 908, 921, 954, 991, 998, 1006, 1054, 1093, 1103, 1108, 1113, 1119, 1134, 1136, 1140, 1146, 1149, 1155, 1161,1226, 1230, 1248, 1255, 1275, 1283, 1289, 1301, 1319, 1330, 1345, 1365, 1385, 1436, 1439, 1447, 1468, 1472, 1566, 1592, 1598, 1617, 1621, 1653, 1680, 1693, 1703, 1705, 1712, 1713, 1721, 1729, 1762, 1764, 1796, 1821, 1822, 1826, 1863, 1873],
	[19, 20, 43, 50, 97, 134, 138, 140, 144, 148, 164, 215, 237, 251, 278, 291, 321, 341, 346, 356, 368, 384, 389, 394, 407, 426, 427, 464, 469, 480, 488, 513, 539, 558, 602, 610, 647, 655, 668, 676, 710, 713, 732, 771, 818, 826, 834, 854, 858, 861, 903, 925, 933, 948, 950, 952, 959, 988, 1000, 1073, 1106, 1121, 1122, 1127, 1137, 1141, 1182, 1183, 1186, 1188, 1191, 1211, 1212, 1218, 1251, 1277, 1280,1290, 1293, 1357, 1368, 1371, 1400, 1453, 1462, 1467, 1477, 1497, 1522, 1558, 1571, 1587, 1710, 1711, 1768, 1780, 1801, 1812, 1815, 1853], 
	[4, 14, 28, 38, 60,75, 81, 105, 117, 135, 161, 174, 181, 194, 216, 222, 239, 266, 273, 275, 299, 302, 318, 329, 344, 350, 374, 385, 413, 441, 455, 460, 495, 542, 612, 645, 650, 674, 689, 691, 785, 803, 807, 825, 885, 907, 930, 945, 968, 980, 989, 993, 1004, 1015, 1025, 1042, 1075, 1084, 1124, 1157, 1168, 1180, 1192, 1213, 1233, 1243, 1257, 1298, 1311, 1328, 1350, 1353, 1401, 1402, 1463, 1508, 1511, 1512, 1541, 1544, 1560, 1570, 1579, 1581, 1593, 1618, 1622, 1640, 1642, 1643, 1670, 1698, 1734, 1758, 1833, 1836, 1845, 1851, 1866, 1895], 
	[0, 35, 40, 53, 55, 77, 82, 83, 89, 156, 172, 203, 209, 234, 242, 256, 295, 337, 338, 369, 401, 428, 435, 436, 459, 463, 468, 475, 477, 508, 530, 572, 574, 583, 621, 648, 649, 652, 687, 701, 705, 712, 729, 749, 780, 788, 805, 809, 836, 860, 862, 919, 987, 1003, 1019, 1047, 1048, 1053, 1139, 1175, 1197, 1209, 1235, 1244, 1246, 1267, 1317, 1332, 1393, 1397,1403, 1411, 1423, 1452, 1454, 1510, 1529, 1537, 1540, 1545, 1574, 1576, 1597, 1608, 1658, 1662, 1681, 1692, 1728, 1732, 1746, 1760, 1763, 1791, 1824, 1825, 1830, 1837, 1874, 1882], 
	[39, 42, 65, 78, 154, 228, 253, 290, 294, 312, 353, 364, 366, 379, 399, 408, 425, 454, 457, 476, 519, 521, 575, 587, 629, 666, 692, 699, 709, 719, 752, 753, 760, 782, 784, 786, 791, 842, 845, 851, 889, 896, 897, 898, 915, 958, 971, 976, 981, 1002, 1009, 1036, 1037, 1043, 1088, 1096, 1099, 1173, 1189, 1201, 1208, 1238, 1279, 1291, 1309, 1315, 1355, 1362, 1376, 1399, 1407, 1433, 1435, 1438, 1458, 1466, 1488, 1493, 1496, 1499, 1513, 1517, 1528, 1542, 1555, 1561, 1626, 1636, 1650, 1676, 1719, 1722, 1731, 1735, 1759, 1765, 1808, 1810, 1888, 1899], 
	[56, 64, 70, 88, 114, 125, 130, 145, 155, 157, 160, 163, 200, 220, 225, 231, 261, 282, 297, 319, 328, 343, 359, 386, 392, 393, 400, 505, 510, 538, 540, 577, 609, 651, 683, 686, 734, 741, 776, 790, 796, 813, 840, 841, 883, 920, 956, 957, 1031, 1033, 1085, 1091, 1101, 1130, 1144, 1148, 1165, 1177, 1190, 1194, 1217, 1222, 1281, 1294, 1321, 1329, 1367, 1373, 1380, 1398, 1408, 1414, 1441, 1450, 1455, 1469, 1484, 1526, 1530, 1594, 1595, 1641, 1660, 1672, 1679, 1682, 1700, 1714, 1720, 1737, 1742, 1777, 1789, 1792, 1795, 1850, 1854, 1871, 1872, 1878],
	[1, 17, 27, 29, 58, 66, 79, 173, 175, 202, 240, 285, 323, 351, 391, 411, 423, 424, 440, 456, 465, 484, 522, 533, 559, 580, 585, 596, 597, 627, 635, 640, 642, 653, 660, 669, 702, 716, 751, 757, 759, 778, 833, 835, 848, 859, 869, 895, 916, 943, 949, 972, 986, 1001, 1032, 1068, 1118, 1123, 1150, 1153, 1195, 1262, 1269, 1273, 1286, 1295, 1296, 1304, 1308, 1313, 1325, 1338, 1340, 1343, 1348, 1372, 1390, 1427, 1440, 1448, 1523, 1531, 1534, 1535, 1554, 1567, 1599, 1605, 1606, 1652,1678, 1687, 1707, 1745, 1747, 1756, 1772, 1778, 1787, 1893], 
	[31, 52, 141, 153,158, 185, 186, 204, 236, 246, 264, 268, 274, 292, 309, 330, 370, 383, 433, 437,458, 481, 496, 517, 520, 555, 557, 570, 591, 595, 601, 614, 623, 630, 659, 681,726, 735, 742, 748, 767, 772, 792, 844, 846, 887, 904, 914, 929, 935, 940, 947,960, 966, 975, 1018, 1041, 1044, 1045, 1098, 1102, 1138, 1162, 1171, 1172, 1229, 1245, 1259, 1268, 1285, 1310, 1337, 1369, 1370, 1382, 1388, 1395, 1412, 1425, 1426, 1437, 1446, 1471, 1479, 1520, 1546, 1604, 1614, 1627, 1646, 1648, 1663, 1665, 1736, 1738, 1749, 1798, 1804, 1817, 1838], 
	[2, 3, 16, 18, 24, 25, 96, 109, 133, 177, 182, 195, 210, 218, 226, 230, 244, 260, 276, 326, 361, 418, 453, 461, 467, 489, 497, 507, 512, 536, 544, 579, 588, 598, 617, 671, 680, 721, 739, 793, 808, 871, 931, 934, 964, 965, 990, 1005, 1023, 1059, 1060, 1063, 1067, 1076, 1154, 1164, 1166, 1205, 1214, 1219, 1232, 1253, 1260, 1261, 1274, 1284, 1292, 1299, 1336, 1344, 1352, 1377, 1404, 1418, 1431, 1445, 1451, 1460, 1486, 1489, 1501, 1556, 1557, 1590, 1644, 1675, 1685, 1744, 1774, 1786, 1788, 1802, 1807, 1811, 1827, 1834, 1835, 1876, 1886, 1894]]

	maxIntersection = 0
	for cluster in clusteringResults:
		finalCount = 0
		for oracleCluster in crossDatasetTrueClassification:
			tempcount = len(set(oracleCluster).intersection(set(cluster)))
			if finalCount < tempcount:
				finalCount = tempcount
		maxIntersection += finalCount
			

	print maxIntersection/19.0

	fileContainer.write('\n')
	fileContainer.write('The clustering Jaccard Similarity is : '+ str(maxIntersection/19.0))


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
			if ii == 19:
				flag = False
		return (jaccard/19.0)*100


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
			if ii == 19:
				flag = False
		return (maxIntersection/1900.0)*100
			

	print jaccardSim(crossDatasetTrueClassification, clusteringResults)

	print ccr(crossDatasetTrueClassification, clusteringResults)
	# print labelAssignment	
	fileContainer.write('\n')
	fileContainer.write('The Correct Clustering Rate is : '+ str(ccr(crossDatasetTrueClassification, clusteringResults)))
	fileContainer.write('\n')
	fileContainer.write('The clustering Jaccard Similarity is : '+ str(jaccardSim(crossDatasetTrueClassification, clusteringResults)))


	wholeTrueClasses = np.zeros(1900)
	for i, trueClass in enumerate(crossDatasetTrueClassification):
		for trajectory in trueClass:
			wholeTrueClasses[trueClass] = i
	print list(wholeTrueClasses[:20])

	wholePredClasses = np.zeros(1900)
	for i, predClass in enumerate(clusteringResults):
		for trajectory in predClass:
			wholePredClasses[predClass] = i

	print list(wholePredClasses[:20])

	print normalized_mutual_info_score(wholeTrueClasses, wholePredClasses)
	fileContainer.write('\nThe NMI is : '+ str(normalized_mutual_info_score(wholeTrueClasses, wholePredClasses)))
	fileContainer.write('\n--------------------------------------------------------------------')

	print measurements.ccr(crossDatasetTrueClassification, clusteringResults)
	print measurements.jaccardSim(crossDatasetTrueClassification, clusteringResults)
	print measurements.NMI(crossDatasetTrueClassification, clusteringResults)
	allCCR.append(measurements.ccr(crossDatasetTrueClassification, clusteringResults))
	allJaccardSim.append(measurements.jaccardSim(crossDatasetTrueClassification, clusteringResults))
	allNMI.append(measurements.NMI(crossDatasetTrueClassification, clusteringResults))
fileContainer.write('\nallCCR\n')
fileContainer.write(str(allCCR))
fileContainer.write('\nallJaccardSim\n')
fileContainer.write(str(allJaccardSim))
fileContainer.write('\nallNMI\n')
fileContainer.write(str(allNMI))
fileContainer.write('\nMIN\nallCCR\n')
fileContainer.write(str(np.min(allCCR)))
fileContainer.write('\nallJaccardSim\n')
fileContainer.write(str(np.min(allJaccardSim)))
fileContainer.write('\nallNMI\n')
fileContainer.write(str(np.min(allNMI)))
fileContainer.write('\nMAX\nallCCR\n')
fileContainer.write(str(np.max(allCCR)))
fileContainer.write('\nallJaccardSim\n')
fileContainer.write(str(np.max(allJaccardSim)))
fileContainer.write('\nallNMI\n')
fileContainer.write(str(np.max(allNMI)))
fileContainer.write('\nAVERAGES\nallCCR\n')
fileContainer.write(str(np.average(allCCR)))
fileContainer.write('\nallJaccardSim\n')
fileContainer.write(str(np.average(allJaccardSim)))
fileContainer.write('\nallNMI\n')
fileContainer.write(str(np.average(allNMI)))
print allCCR
print allJaccardSim
print allNMI
print 'MINs'
print np.min(allCCR)
print np.min(allJaccardSim)
print np.min(allNMI)
print 'MAXs'
print np.max(allCCR)
print np.max(allJaccardSim)
print np.max(allNMI)
print 'AVERAGEs'
print np.average(allCCR)
print np.average(allJaccardSim)
print np.average(allNMI)
