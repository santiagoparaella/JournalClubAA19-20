#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn import datasets
from cknn import cknneighbors_graph
import scipy
import scipy.sparse as sparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import random as rand
import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
import markov_clustering as mc

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics.pairwise import pairwise_distances
from graphs import Graph

import networkx as nx 
from sklearn.cluster import SpectralClustering


def connect_points(ax, data, graph):
    source, target = graph.nonzero()
    source, target = source[source < target], target[source < target]
    for s, t in zip(source, target):
        ax.plot(*data[[s,t], :].T, color='g')

def plot_graph(data, graph, title=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    if title is not None:
        ax.set_title(title)
    ax.scatter(*data.T)
    connect_points(ax, data, graph)
    plt.show()
    plt.close()


def cknneighbors_graph(X, n_neighbors, delta=1.0, metric='euclidean', t='inf',
                       include_self=False, is_sparse=True,
                       return_instance=False):

    cknn = CkNearestNeighbors(n_neighbors=n_neighbors, delta=delta,
                              metric=metric, t=t, include_self=include_self,
                              is_sparse=is_sparse)
    cknn.cknneighbors_graph(X)

    if return_instance:
        return cknn
    else:
        return cknn.ckng


class CkNearestNeighbors(object):
    """This object provides the all logic of CkNN.
    Args:
        n_neighbors: int, optional, default=5
            Number of neighbors to estimate the density around the point.
            It appeared as a parameter `k` in the paper.
        delta: float, optional, default=1.0
            A parameter to decide the radius for each points. The combination
            radius increases in proportion to this parameter.
        metric: str, optional, default='euclidean'
            The metric of each points. This parameter depends on the parameter
            `metric` of scipy.spatial.distance.pdist.
        t: 'inf' or float or int, optional, default='inf'
            The decay parameter of heat kernel. The weights are calculated as
            follow:
                W_{ij} = exp(-(||x_{i}-x_{j}||^2)/t)
            For more infomation, read the paper 'Laplacian Eigenmaps for
            Dimensionality Reduction and Data Representation', Belkin, et. al.
        include_self: bool, optional, default=True
            All diagonal elements are 1.0 if this parameter is True.
        is_sparse: bool, optional, default=True
            The method `cknneighbors_graph` returns csr_matrix object if this
            parameter is True else returns ndarray object.
    """

    def __init__(self, n_neighbors=5, delta=1.0, metric='euclidean', t='inf',
                 include_self=False, is_sparse=True):
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.metric = metric
        self.t = t
        self.include_self = include_self
        self.is_sparse = is_sparse
        self.ckng = None

    def cknneighbors_graph(self, X):
        """A method to calculate the CkNN graph
        Args:
            X: ndarray
                The data matrix.
        return: csr_matrix (if self.is_sparse is True)
                or ndarray(if self.is_sparse is False)
        """

        n_neighbors = self.n_neighbors
        delta = self.delta
        metric = self.metric
        t = self.t
        include_self = self.include_self
        is_sparse = self.is_sparse

        n_samples = X.shape[0]

        if n_neighbors < 1 or n_neighbors > n_samples-1:
            raise ValueError("`n_neighbors` must be "
                             "in the range 1 to number of samples")
        if len(X.shape) != 2:
            raise ValueError("`X` must be 2D matrix")
        if n_samples < 2:
            raise ValueError("At least 2 data points are required")

        if metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("`X` must be square matrix")
            dmatrix = X
        else:
            dmatrix = squareform(pdist(X, metric=metric))

        darray_n_nbrs = np.partition(dmatrix, n_neighbors)[:, [n_neighbors]]
        ratio_matrix = dmatrix / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
        diag_ptr = np.arange(n_samples)

        if isinstance(delta, (int, float)):
            ValueError("Invalid argument type. "
                       "Type of `delta` must be float or int")
        adjacency = csr_matrix(ratio_matrix < delta)

        if include_self:
            adjacency[diag_ptr, diag_ptr] = True
        else:
            adjacency[diag_ptr, diag_ptr] = False

        if t == 'inf':
            neigh = adjacency.astype(np.float)
        else:
            mask = adjacency.nonzero()
            weights = np.exp(-np.power(dmatrix[mask], 2)/t)
            dmatrix[:] = 0.
            dmatrix[mask] = weights
            neigh = csr_matrix(dmatrix)

        if is_sparse:
            self.ckng = neigh
        else:
            self.ckng = neigh.toarray()

        return self.ckng
def mst(X, metric='euclidean'):
    D = pairwise_distances(X, metric=metric)
    mst = minimum_spanning_tree(D, overwrite=(metric!='precomputed'))
    return Graph.from_adj_matrix(mst + mst.T)


def perturbed_mst(X, num_perturbations=20, metric='euclidean', jitter=None):
    '''Builds a graph as the union of several MSTs on perturbed data.
    Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 8
    jitter refers to the scale of the gaussian noise added for each perturbation.
    When jitter is None, it defaults to the 5th percentile interpoint distance.
    Note that metric cannot be 'precomputed', as multiple MSTs are computed.'''
    assert metric != 'precomputed'
    D = pairwise_distances(X, metric=metric)
    if jitter is None:
        jitter = np.percentile(D[D>0], 5)
    W = minimum_spanning_tree(D)
    W = W + W.T
    W.data[:] = 1.0  # binarize
    for i in range(num_perturbations):
        pX = X + np.random.normal(scale=jitter, size=X.shape)
        pW = minimum_spanning_tree(pairwise_distances(pX, metric=metric))
        pW = pW + pW.T
        pW.data[:] = 1.0
        W = W + pW
    # final graph is the average over all pertubed MSTs + the original
    W.data /= (num_perturbations + 1.0)
    return Graph.from_adj_matrix(W)

