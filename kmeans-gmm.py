import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import random as rand
from scipy.stats import norm
from sys import maxint
import math

colorlist = ['g', 'b', 'c']

#Function to convert the pandas data to float
def toFloat(data, attrs):
    if type(attrs) == list:
        for attr in attrs:
            data[attr] = data[attr].astype(float)
    return data


#Function to calculate the Euclidean distance
def dist(a, b, ax=1):
    distance = np.linalg.norm(a - b, axis=ax)
    return distance


def kmeans(data, k):

    x1 = data['x1'].values
    x2 = data['x2'].values
    X = np.array(list(zip(x1, x2)))
    plt.title('K Means - Initial Centroids')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x1, x2, c='black', s=10)

    #Initializing random values for centroids
    Cx = np.random.randint(0, np.max(X), size=k)
    Cy = np.random.randint(0, np.max(X), size=k)
    C = np.array(list(zip(Cx, Cy)), dtype=np.float32)

    #Plotting the data with initial centroids
    plt.scatter(x1, x2, c='black', s=10)
    plt.scatter(Cx, Cy, marker='D', s=50, c='r')

    #Storing the values of Centroids before updating
    Cprev = np.zeros(C.shape)
    clusters = np.zeros(len(X))
    #Calculating the Euclidean distance between the previous Centroid values and current Centroid values
    err = dist(C, Cprev, None)

    while err != 0:
        #Allocating each point to its closest cluster
        for i in range(len(X)):
            distance = dist(X[i], C)
            cluster = np.argmin(distance)
            clusters[i] = cluster
        #Storing the values of Centroids before updating
        Cprev = deepcopy(C)
        #Calculating new values for Centroids
        for i in range(k):
            points = []    
            for j in range(len(X)):
               if clusters[j] == i:
                    points.append(X[j]) 
            C[i] = np.mean(points, axis=0)
        #Calculating the Euclidean distance between the previous Centroid values and current Centroid values
        err = dist(C, Cprev, None)

    #Plotting the final Centroids with the final Clusters
    Cov = []
    fig, ax = plt.subplots()
    plt.title('K Means - Final Clusters')
    plt.xlabel('x1')
    plt.ylabel('x2')
    for i in range(k):
        points = []   
        for j in range(len(X)):
            if clusters[j] == i:
                points.append(X[j]) 
        points = np.array(points)
        #Calculating covariance of each cluster
        Cov.append(np.cov(data, rowvar=False))
        ax.scatter(points[:, 0], points[:, 1], s=10, c=colorlist[i])
    print Cov
    ax.scatter(C[:, 0], C[:, 1], marker='D', s=50, c='r')
    plt.show()
    C = np.array(C).tolist()
    Cov = np.array(Cov).tolist()
    print C
    print Cov
    return (C, Cov)


#Probability of a point coming from a given Gaussian
def prob(val, mu, sig, lam):
  p = lam
  for i in range(len(val)):
    p *= norm.pdf(val[i], mu[i], sig[i][i])
  return p


#Expectation
def Expectation(dataFrame, param):
  for i in range(dataFrame.shape[0]):
    x = dataFrame['x'][i]
    y = dataFrame['y'][i]
    prob_clstr1 = prob([x, y], list(param['mu1']), list(param['sig1']), param['lambda'][0][0])
    prob_clstr2 = prob([x, y], list(param['mu2']), list(param['sig2']), param['lambda'][0][1])
    prob_clstr3 = prob([x, y], list(param['mu3']), list(param['sig3']), param['lambda'][0][2])
    if prob_clstr1 > prob_clstr2 and prob_clstr1 > prob_clstr3:
      dataFrame['color'][i] = 'g'
    elif prob_clstr2 > prob_clstr1 and prob_clstr2 > prob_clstr3:
      dataFrame['color'][i] = 'b'
    else:
      dataFrame['color'][i] = 'c'
  return dataFrame


#Maximization
def Maximization(dataFrame, param):
  points_clstr1 = dataFrame[dataFrame['color'] == 'g']
  points_clstr2 = dataFrame[dataFrame['color'] == 'b']
  points_clstr3 = dataFrame[dataFrame['color'] == 'c']
  percent_clstr1 = len(points_clstr1)/float(len(dataFrame))
  percent_clstr2 = len(points_clstr2)/float(len(dataFrame))
  percent_clstr3 = len(points_clstr3)/float(len(dataFrame))
  param['lambda'] = [[percent_clstr1, percent_clstr2, percent_clstr3], [0]]
  param['mu1'] = [points_clstr1['x'].mean(), points_clstr1['y'].mean()]
  param['mu2'] = [points_clstr2['x'].mean(), points_clstr2['y'].mean()]
  param['mu3'] = [points_clstr3['x'].mean(), points_clstr3['y'].mean()]
  param['sig1'] = [ [points_clstr1['x'].std(), 0 ], [ 0, points_clstr1['y'].std() ] ]
  param['sig2'] = [ [points_clstr2['x'].std(), 0 ], [ 0, points_clstr2['y'].std() ] ]
  param['sig3'] = [ [points_clstr3['x'].std(), 0 ], [ 0, points_clstr3['y'].std() ] ]
  return param

#Function to calculate the Euclidean distance between points
def distance(params, newparams):
  dist = 0
  for param in ['mu1', 'mu2', 'mu3']:
    for i in range(len(params)):
      dist += math.pow((params[param][i] - newparams[param][i]), 2)
  return math.sqrt(dist)


def GMM(dataFrame, C, Cov):
    xs = dataFrame['x1'].values
    ys = dataFrame['x2'].values

    colors = (['g'] * 1000) + (['b'] * 1000) + (['c'] * 1000)

    data = {'x': xs, 'y': ys, 'color': colors}
    df = pd.DataFrame(data=data)

    fig = plt.figure()
    plt.title('GMM - Initial')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(data['x'], data['y'], c=data['color'])
    plt.show()

    #Initialising mean and covariance from kmeans algorithm
    init = {'mu1': C[0], 'sig1': Cov[0], 'mu2': C[1], 'sig2': Cov[1], 'mu3': C[2], 'sig3': Cov[2], 'lambda': [[0.33, 0.33, 0.33], [0]]}

    err = maxint
    it = 0

    #Initially assigning points to clusters at random
    df['color'] = rand.choice(colorlist)
    params = pd.DataFrame(init)

    while err != 0:
        it += 1

        newlabels = Expectation(df, params)
        newparams = Maximization(newlabels, params.copy())

        #Calculating the Euclidean distance to check for change
        err = distance(params, newparams)

        #Displaying iteration number and error
        print("Iteration number {} with error {}".format(it, err))

        #Updating with new labels and parameters for the next iteration
        df = newlabels
        params = newparams

        fig = plt.figure()
        plt.title('GMM - Iteration %d' %it)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.scatter(df['x'], df['y'], c=df['color'])
        plt.show()
    return

dataFrame = pd.read_csv('xclara_data.csv')
dataFrame = toFloat(dataFrame, ['x1', 'x2'])

mean_cov = kmeans(dataFrame, 3)
GMM(dataFrame, mean_cov[0], mean_cov[1])