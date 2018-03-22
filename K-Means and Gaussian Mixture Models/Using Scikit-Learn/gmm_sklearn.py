from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
from sklearn import mixture

def GMM_sklearn(dataFrame):
	x1 = data['x1'].values
	x2 = data['x2'].values
	X = np.array(list(zip(x1, x2)))
	gmix = mixture.GaussianMixture(n_components=3, init_params='kmeans').fit(X)
	print gmix.means_
	print "Number of iterations: %d" %gmix.n_iter_

	color = []
	for i in gmix.predict(X):
		if i == 0:
			color.append('g')
		elif i == 1:
			color.append('b')
		else:
			color.append('c')

	plt.title('GMM - sklearn')
	plt.xlabel('x1')
	plt.ylabel('x2')
	ax = plt.gca()
	ax.scatter(X[:,0], X[:,1], c=color)
	plt.show()

colors = ['g', 'b', 'c']
data = pd.read_csv('xclara_data.csv')

GMM_sklearn(data)
