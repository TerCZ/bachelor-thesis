import time

import numpy as np


def sample(X, y, K):
    sample_X = []
    sample_y = []

    max_X = np.max(X)
    min_X = np.min(X)

    i = 0
    last_X = X[0]
    last_y = y[0]
    for point_i in range(len(X)):
        desired_X = i / K * (max_X - min_X) + min_X

        if X[point_i] < desired_X:
            last_X = X[point_i]
            last_y = y[point_i]
            continue

        if y[point_i] - last_y == 0:
            estimated_y = 0
        else:
            estimated_y = y[point_i] - (X[point_i] - desired_X) / \
                (X[point_i] - last_X) * (y[point_i] - last_y)
        sample_X.append(desired_X)
        sample_y.append(estimated_y)

        last_X = X[point_i]
        last_y = y[point_i]
        i += 1
        if i == K:
            break

    sample_X = np.asarray(sample_X)
    sample_y = np.asarray(sample_y)
    sample_X = (sample_X - np.min(sample_X)) / \
        (np.max(sample_X) - np.min(sample_X))
    sample_y = (sample_y - np.min(sample_y)) / \
        (np.max(sample_y) - np.min(sample_y))
    return sample_X, sample_y


datasets = [("../data/n10-1682-X-small.npy", "../data/n10-1682-y-small.npy"),
            ("../data/n10-3127-X-small.npy", "../data/n10-3127-y-small.npy"),
            ("../data/n10-6797-X-small.npy", "../data/n10-6797-y-small.npy"),
            ("../data/n20-6174-X-small.npy", "../data/n20-6174-y-small.npy")]
raws = []
for ds in datasets:
    X = np.load(ds[0])
    y = np.load(ds[1])
    raws.append((X, y))


K = 100
data = []
for X, y in raws:
    data.append(sample(X, y, K))


for ds_1 in data:
    for ds_2 in data:
        similarity = sum([(ds_1[1][i] - ds_2[1][i]) **
                          2 for i in range(len(ds_1[1]))]) / K
        print("%.2f" % similarity, end="\t")
    print()


for K in [100, 500, 1000, 5000]:
    for entry_n in [1000, 5000, 10000, 50000, 100000]:
        start = time.time()
        sample_X, sample_y = sample(raws[0][0], raws[0][1], K)
        for _ in range(entry_n):
            (sample_y - sample_y) ** 2 / K
        print("%.2f" % (time.time() - start), end="  & ")
    print()