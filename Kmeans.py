__authors__ = ['1752744', '1750906']
__group__ = '48'

import numpy as np
import utils
import random
from scipy.spatial.distance import cdist


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        X = np.array(X)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[2])
        self.X = X.astype(float)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.centroids = np.random.rand(self.K, self.X.shape[1])
        self.old_centroids = np.random.rand(self.K, self.X.shape[1])


        if self.options['km_init'].lower() == 'first':
            #Codi que no funciona del tot bé
            
            anteriors = []
            k = 0

            for i in range(0, self.X.shape[0]):
                if k < self.K:
                    actual = []
                    for j in range(0, 3):
                        actual.append(self.X[i][j])
                    if actual not in anteriors:
                        for j in range(0, 3):
                            self.centroids[k][j] = self.X[i][j]
                        k += 1
                        anteriors.append(actual)
        else:
            nPixels = self.X.shape[0]
            for i in range(self.K):
                pixel = random.randint(0, nPixels - 1)
                self.centroids[i] = self.X[pixel]


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """

        #self.labels = np.random.randint(self.K, size=self.X.shape[0])

        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

        #self.labels = np.zeros(self.X.shape[0]).astype(int)

        #for i in range(0, self.X.shape[0]):
        #    self.labels[i] = dist[i].argmin()
            

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids = self.centroids.copy()

        #for i in range(0, self.K):
        #    list = []
        #    for j in range(0, self.labels.shape[0]):
        #        if self.labels[j] == i:
        #            list.append(self.X[j])

        #    points = np.array(list, dtype=float)
        #    self.centroids[i] = np.mean(points, axis=0)
        for i in range(self.K):
            points = self.X[self.labels == i]
            if len(points) > 0:
                self.centroids[i] = points.mean(axis=0)



    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        #return self.centroids.all() != self.old_centroids.all()
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """

        self._init_centroids()
        self.num_iter = 0
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            #print(self.centroids)

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        distances = 0
    
        #for i in range(0, self.X.shape[0]):
        #    point = self.X[i].astype(float)
        #    centroid = self.centroids[int(self.labels[i])].astype(float)

        #    distances += np.sum((point - centroid) ** 2)

        #self.wcd = distances / len(self.X)
        diff = self.X - self.centroids[self.labels]
        self.wcd = np.mean(np.sum(diff ** 2, axis=1))


        #print("\n Average:", average, "\n")

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """

        wcd_list = []

        for i in range(1, max_K + 1):
            self.K = i
            self.fit()
            self.withinClassDistance()
            wcd_list.append(self.wcd)
        #print(wcd_list)
        for i in range(1, max_K):
            decrease = 100 * (wcd_list[i]/wcd_list[i - 1])
            if (100 - decrease) < 20:
                self.K = i
                self.fit()
                return
        self.K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    return cdist(X, C)

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    labels = []
    probs = utils.get_color_prob(centroids)
    indexs = probs.argmax(axis = 1)
    labels = list(utils.colors[indexs])
    return labels
