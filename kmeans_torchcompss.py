from pycompss.api.task import task
from pycompss.api.parameter import * 
import torch
import numpy as np


@task(returns=torch.DoubleTensor)
def sum_centroids(a, b):
    return a+b


def mergeReduce(function, data):
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = function(data[x], data[y])
            q.append(x)
        else:
            return data[x]


@task(returns=torch.DoubleTensor)
def assign(vectors, assignments, k, dim):
    centroids = []
    for c in range(k):
        to_cat = torch.unsqueeze(assignments == c, 1)
        if dim == 1:
            concat = to_cat
        else:
            concat = torch.cat((to_cat, to_cat), 1)
            for j in range(dim-2):
                concat = torch.cat((concat, to_cat), 1)
        selected = torch.masked_select(vectors, concat).view(-1, dim)
        centroids.append(torch.mean(selected, 0).view(1, dim))
    return torch.cat(centroids, 0)


@task(returns=torch.LongTensor)
def kmeans_torch(expanded_vectors, centroids):
    expanded_centroids = torch.unsqueeze(centroids, 1)
    distances = torch.cumsum((expanded_vectors - expanded_centroids)**2, 2)[:, :, 1]
    values, assignments = torch.min(distances, 0)
    return assignments


@task(returns=torch.DoubleTensor)
def genFragments(size, dim):
    vectors_set = []
    for i in range(size):
        if np.random.random() > 0.5:
            point = [np.random.normal(0.0, 0.9) for _ in range(dim)]
        else:
            point = [np.random.normal(1.5, 0.5) for _ in range(dim)]
        vectors_set.append(point)
    return torch.from_numpy(np.array(vectors_set))


@task(returns=torch.DoubleTensor)
def unsqueeze(x):
    return torch.unsqueeze(x, 0)


def kmeans_frag(numP, k, dim, convergenceFactor, maxIterations, numFrag):
    from pycompss.api.api import compss_wait_on
    import time

    startTime = time.time()

    size = int(numP/numFrag)
    X = [genFragments(size, dim) for _ in range(numFrag)]
    X_e = [unsqueeze(X[i]) for i in range(numFrag)]

    centroids = genFragments(k, dim)
    for n in range(maxIterations):
        assignments = [kmeans_torch(X_e[i], centroids) for i in range(numFrag)]
        centroids_partial = [assign(X[i], assignments[i], k, dim) for i in range(numFrag)]
        new_centroids = mergeReduce(sum_centroids, centroids_partial)
        new_centroids = compss_wait_on(new_centroids)
        centroids = torch.div(new_centroids, numFrag)

    print("Kmeans Time "+str(time.time() - startTime)+" (s)")


if __name__ == "__main__":
    import sys
    import time

    numP = int(sys.argv[1])
    dim = int(sys.argv[2])
    k = int(sys.argv[3])
    numFrag = int(sys.argv[4])

    startTime = time.time()
    kmeans_frag(numP, k, dim, 1e-4, 5, numFrag)
    print("Ellapsed Time "+str(time.time() - startTime)+" (s)")
