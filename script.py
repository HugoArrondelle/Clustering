

from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import pandas as pd 
import numpy as np
import sys, getopt
import hdbscan




def compute(algoName,fileName, metricsName, linkageName, nbClusterStart, nbClusterEnd, nbEspStart, nbEspEnd, nbMinSampleStart, nbMinSampleEnd):
	data = np.loadtxt(open(fileName,'r'))

	computedAlgo=0


	if(algoName=='kmeans'):
		nbCluster = findBestClusterNumber(algoName, metricsName, '', data,nbClusterStart,nbClusterEnd, 0, 0, 0, 0)
		computedAlgo = KMeans(n_clusters=nbCluster, init='k-means++').fit(data)
	elif(algoName=='agglo'):
		nbCluster = findBestClusterNumber(algoName, metricsName, linkageName, data,nbClusterStart,nbClusterEnd, 0, 0, 0, 0)		
		computedAlgo = AgglomerativeClustering(n_clusters=nbCluster, linkage=linkageName).fit(data)
	elif(algoName=='dbscan'):
		nbEsp, nbMinSample = findBestClusterNumber(algoName, metricsName, '', data, 0, 0, nbEspStart, nbEspEnd, nbMinSampleStart, nbMinSampleEnd)		
		computedAlgo = DBSCAN(eps=nbEsp, min_samples=nbMinSample).fit(data)
	elif(algoName=='hdbscan'):
		nbMinSample = findBestClusterNumber(algoName, metricsName, '', data, 0, 0, 0, 0, nbMinSampleStart, nbMinSampleEnd)		
		computedAlgo = hdbscan.HDBSCAN(min_samples = nbMinSample).fit(data)

	if(data.shape[1] == 2):
		plt.scatter(data[:,0], data[:,1],s=10, c=computedAlgo.labels_)
		plt.show()
	elif(data.shape[1] == 3):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter3D(data[:,0], data[:,1], data[:,2], c=computedAlgo.labels_)
		plt.show()
	else:
		print("ERROR DATASET")


def findBestClusterNumber(algoName, metricsName, linkageName, data, nbClusterStart, nbClusterEnd, nbEspStart, nbEspEnd, nbMinSampleStart, nbMinSampleEnd):
	
	nbClusterList = [i for i in range(nbClusterStart, nbClusterEnd+1)]
	bestMatch_silhouette = -float('inf')
	bestMatch_davies_bouldin = float('inf')
	bestMatch_calinski_harabasz = -float('inf')
	index_best = -1

	if(algoName=='kmeans'):
		for i,nb_cluster in enumerate(nbClusterList):
			Kmeans = KMeans(n_clusters=nb_cluster, init='k-means++').fit(data)
			n_clusters_ = len(set(Kmeans.labels_)) - (1 if -1 in Kmeans.labels_ else 0)
				
			if (n_clusters_ > 1):
				if(metricsName=='silhouette_score'):
					score = metrics.silhouette_score(data, Kmeans.labels_ , metric='euclidean')
					if(bestMatch_silhouette < score):
						bestMatch_silhouette = score 
						index_best = i
				elif(metricsName=='davies_bouldin_score'):
					score = metrics.davies_bouldin_score(data, Kmeans.labels_)
					if(bestMatch_davies_bouldin > score):
						bestMatch_davies_bouldin = score 
						index_best = i
				elif(metricsName=='calinski_harabasz_score'):
					score = metrics.calinski_harabasz_score(data, Kmeans.labels_)
					if(bestMatch_calinski_harabasz < score):
						bestMatch_calinski_harabasz = score 
						index_best = i
		return nbClusterList[index_best]

	if(algoName=='agglo'):
		for i,nb_cluster in enumerate(nbClusterList):
			agglo = AgglomerativeClustering(n_clusters=nb_cluster, linkage=linkageName).fit(data)
			n_clusters_ = len(set(agglo.labels_)) - (1 if -1 in agglo.labels_ else 0)
				
			if (n_clusters_ > 1):
				if(metricsName=='silhouette_score'):
					score = metrics.silhouette_score(data, agglo.labels_ , metric='euclidean')
					if(bestMatch_silhouette < score):
						bestMatch_silhouette = score 
						index_best = i
				elif(metricsName=='davies_bouldin_score'):
					score = metrics.davies_bouldin_score(data, agglo.labels_)
					if(bestMatch_davies_bouldin > score):
						bestMatch_davies_bouldin = score 
						index_best = i
				elif(metricsName=='calinski_harabasz_score'):
					score = metrics.calinski_harabasz_score(data, agglo.labels_)
					if(bestMatch_calinski_harabasz < score):
						bestMatch_calinski_harabasz = score 
						index_best = i
		return nbClusterList[index_best]

	if(algoName=='dbscan'):
	
		nbEspList = [i for i in np.arange(nbEspStart, nbEspEnd, 0.1)]
		nbMinSampleList = [j for j in range(nbMinSampleStart, nbMinSampleEnd+1)]

		index_best_Esp = -1
		index_best_MinSample = -1

		for i,nb_Esp in enumerate(nbEspList):
			for j,nb_MinSample in enumerate(nbMinSampleList):
				dbscan = DBSCAN(eps=nb_Esp, min_samples=nb_MinSample).fit(data)
				n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
				
				if (n_clusters_ > 1):
					if(metricsName=='silhouette_score'):
						score = metrics.silhouette_score(data, dbscan.labels_ , metric='euclidean')
						if(bestMatch_silhouette < score):
							bestMatch_silhouette = score 
							index_best_Esp = i
							index_best_MinSample = j
					elif(metricsName=='davies_bouldin_score'):
						score = metrics.davies_bouldin_score(data, dbscan.labels_)
						if(bestMatch_davies_bouldin > score):
							bestMatch_davies_bouldin = score 
							index_best_Esp = i
							index_best_MinSample = j
					elif(metricsName=='calinski_harabasz_score'):
						score = metrics.calinski_harabasz_score(data, dbscan.labels_)
						if(bestMatch_calinski_harabasz < score):
							bestMatch_calinski_harabasz = score 
							index_best_Esp = i
							index_best_MinSample = j

		return nbEspList[index_best_Esp],nbMinSampleList[index_best_MinSample]


	if(algoName=='hdbscan'):
		nbMinSampleList = [j for j in range(nbMinSampleStart, nbMinSampleEnd+1)]
		index_best_MinSample = -1

		for j,nb_MinSample in enumerate(nbMinSampleList):
			Hdbscan = hdbscan.HDBSCAN(min_samples = nb_MinSample).fit(data)
			n_clusters_ = len(set(Hdbscan.labels_)) - (1 if -1 in Hdbscan.labels_ else 0)
			if (n_clusters_ > 1):
				if(metricsName=='silhouette_score'):
					score = metrics.silhouette_score(data, Hdbscan.labels_ , metric='euclidean')
					if(bestMatch_silhouette < score):
						bestMatch_silhouette = score 
						index_best_MinSample = j
				elif(metricsName=='davies_bouldin_score'):
					score = metrics.davies_bouldin_score(data, Hdbscan.labels_)
					if(bestMatch_davies_bouldin > score):
						bestMatch_davies_bouldin = score 
						index_best_MinSample = j
				elif(metricsName=='calinski_harabasz_score'):
					score = metrics.calinski_harabasz_score(data, Hdbscan.labels_)
					if(bestMatch_calinski_harabasz < score):
						bestMatch_calinski_harabasz = score 
						index_best_MinSample = j  
		return nbMinSampleList[index_best_MinSample]


def main():
	arguments = len(sys.argv) - 1
	algoName = input ("Entrer le nom de l'algorithme (kmeans, agglo, dbscan, hdbscan) : ")
	fileName = ''
	metricsName = ''
	linkageName = ''
	nbClusterStart = 0
	nbClusterEnd = 0
	nbEspStart = 0
	nbEspEnd = 0
	nbMinSampleStart = 0
	nbMinSampleEnd = 0

	if(algoName == 'kmeans'):
   		fileName = input ("Entrer le chemin du fichier : ")
   		metricsName = input ("Entrer le nom de la metrics (silhouette_score, davies_bouldin_score, calinski_harabasz_score) : ")
   		nbClusterStart = input ("Entrer le nombre de cluster minimum : ")
   		nbClusterEnd = input ("Entrer le nombre de cluster maximum : ")
   		compute(algoName,fileName,metricsName, '', int(nbClusterStart),int(nbClusterEnd), 0, 0, 0, 0)

	elif(algoName == 'agglo'):
		fileName = input ("Entrer le chemin du fichier : ")
		metricsName = input ("Entrer le nom de la metrics (silhouette_score, davies_bouldin_score, calinski_harabasz_score) : ")
		linkageName = input ("Entrer le nom du linkage (single, ward, complete, average) : ")
		nbClusterStart = input ("Entrer le nombre de cluster minimum : ")
		nbClusterEnd = input ("Entrer le nombre de cluster maximum : ")
		compute(algoName,fileName,metricsName, linkageName, int(nbClusterStart),int(nbClusterEnd), 0, 0, 0, 0)

	elif(algoName == 'dbscan'):
		fileName = input ("Entrer le chemin du fichier : ")
		metricsName = input ("Entrer le nom de la metrics (silhouette_score, davies_bouldin_score, calinski_harabasz_score) : ")
		nbEspStart = input ("Entrer la valeur de esp minimum (>0): ")
		nbEspEnd = input ("Entrer la valeur de esp maximum : ")
		nbMinSampleStart = input ("Entrer la valeur de minSample minimum : ")
		nbMinSampleEnd = input ("Entrer la valeur de minSample maximum : ")
		compute(algoName,fileName,metricsName, '', 0, 0, float(nbEspStart), float(nbEspEnd), int(nbMinSampleStart), int(nbMinSampleEnd))

	elif(algoName == 'hdbscan'):
		fileName = input ("Entrer le chemin du fichier : ")
		metricsName = input ("Entrer le nom de la metrics (silhouette_score, davies_bouldin_score, calinski_harabasz_score) : ")
		nbMinSampleStart = input ("Entrer la valeur de minSample minimum : ")
		nbMinSampleEnd = input ("Entrer la valeur de minSample maximum : ")
		compute(algoName,fileName,metricsName, '', 0, 0, 0, 0, int(nbMinSampleStart),int(nbMinSampleEnd))
	else:
		print("Algorithme incorrect")
		exit(2)


if __name__ == '__main__':
	sys.exit(main())


	