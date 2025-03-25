from pathlib import Path
from PIL import Image
import sys
import time

import matplotlib.pyplot as plt

import numpy as np

import scipy
from scipy.optimize import linear_sum_assignment

import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import fetch_olivetti_faces, make_blobs
from sklearn.neighbors import kneighbors_graph

import torchvision.datasets as datasets

# np.random.seed(7404)
# sklearn.random.seed(7404)
np.set_printoptions(threshold=sys.maxsize)


def compute_clustering_accuracy(ground_truth, prediction, c: int):
    dict_gt = {i: set() for i in range(c)}
    dict_pred = {i: set() for i in range(c)}
    for index, cluster_id in enumerate(ground_truth):
        dict_gt[cluster_id].add(index)
    for index, cluster_id in enumerate(prediction):
        dict_pred[cluster_id].add(index)
    # cost matrix
    cost_matrix = np.zeros(shape=(c, c))
    for i in range(c):
        for j in range(c):
            true_pred = dict_pred[i].intersection(dict_gt[j])
            cost_matrix[i, j] = len(dict_pred[i]) - len(true_pred)
    # Hungarian (Kuhn–Munkres) algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    false_count = cost_matrix[row_ind, col_ind].sum()
    return 1 - false_count / len(ground_truth)


def get_dataset(use_data: str, verbose: bool=False):
    if use_data == 'blob':
        num_clusters = 5
        num_samples = 200
        X, y = make_blobs(n_samples=num_samples, n_features=20, centers=num_clusters)
    elif use_data == 'mnist':
        num_clusters = 10

        mnist_testset = datasets.MNIST(root='../mnist_data', train=False, download=True, transform=None)

        num_samples = len(mnist_testset)

        X = mnist_testset.data.numpy().reshape(num_samples, -1) / 255
        y = mnist_testset.targets.numpy()
    elif use_data == 'cifar':
        num_clusters = 10

        cifar_testset = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=None)

        num_samples = len(cifar_testset)

        X = cifar_testset.data.reshape(num_samples, -1) / 255
        y = np.array(cifar_testset.targets)
    elif use_data == 'usps':
        num_clusters = 10

        usps_testset = datasets.USPS(root='../dataset', train=False, download=True, transform=None)

        num_samples = len(usps_testset)

        X = usps_testset.data.reshape(num_samples, -1) / 255
        y = np.array(usps_testset.targets)
    elif use_data == 'coil':
        num_clusters = 20

        coil_image_paths = Path('../dataset/coil-20').glob('**/*.png')
        X = []
        y = []
        for _img_path in coil_image_paths:
            X.append(np.array(Image.open(_img_path)))
            y.append(int(str(_img_path.parent).split('/')[-1]) - 1)

        num_samples = len(X)

        X = np.array(X).reshape(num_samples, -1) / 255
        y = np.array(y)
    elif use_data == 'mpeg':
        num_clusters = 70

        mpeg_image_paths = Path('../dataset/mpeg7').glob('**/*.gif')
        mpeg_classes = {}
        X = []
        y = []

        for _img_path in mpeg_image_paths:
            _img = np.array(Image.open(_img_path).resize((256, 256)))
            if _img.max() > 1:
                _img = _img / 255
            X.append(_img)
            mpeg_class_name = str(_img_path.parent).split('/')[-1]
            if mpeg_class_name not in mpeg_classes:
                mpeg_classes[mpeg_class_name] = len(mpeg_classes)
            y.append(mpeg_classes[mpeg_class_name])

        num_samples = len(X)

        X = np.array(X).reshape(num_samples, -1)
        y = np.array(y)
    elif use_data == 'olivetti':
        num_clusters = 40

        dataset = fetch_olivetti_faces()
        olive_images = dataset.images
        olive_labels = dataset.target

        num_samples = olive_images.shape[0]

        X = olive_images.reshape(num_samples, -1)
        y = olive_labels
    else:
        assert False
    # ================================================================================
    if verbose:
        print(
            f'''Training dataset {use_data}:
    X shape:   {X.shape}
    y shape: {y.shape}'''
        )
    return X, y, num_clusters


def spectral_clustering(
        X,
        num_clusters,
        num_neighbors: int=10,
        sc_version: str='unnormed',
        drop_first_eigen: bool=None,
):
    assert sc_version in ['unnormed', 'sym-normed', 'rw-normed']

    if drop_first_eigen is None:
        if sc_version in ['sym-normed', 'rw-normed']:
            drop_first_eigen = True
        else:
            drop_first_eigen = False

    # Part 1: similarity graph
    similarity_graph = kneighbors_graph(X, n_neighbors=num_neighbors, include_self=True)
    similarity_matrix = similarity_graph.toarray()
    similarity_matrix[(similarity_matrix + similarity_matrix.T) > 0] = 1

    # Part 2: graph Laplacian
    degree_matrix = np.diag(np.squeeze(np.asarray(np.sum(similarity_matrix, axis=1))))
    laplacian_matrix = degree_matrix - similarity_matrix
    if sc_version == 'unnormed':
        pass
    elif sc_version == 'sym-normed':
        inv_sqrt_degree_matrix = np.linalg.inv(np.sqrt(degree_matrix))
        laplacian_matrix = inv_sqrt_degree_matrix @ laplacian_matrix @ inv_sqrt_degree_matrix
    elif sc_version == 'rw-normed':
        inv_degree_matrix = np.linalg.inv(degree_matrix)
        laplacian_matrix = inv_degree_matrix @ laplacian_matrix
    else:
        assert False

    # Part 3: eigen decomposition
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        -laplacian_matrix,  # for acceleration
        k=num_clusters*2,  # num_cluster is enough for clustering, the remainings are for plot
        sigma=1.0, 
        which='LM'
    )
    if drop_first_eigen:
        k_eigen_matrix = np.flip(eigenvectors, axis=1)[:, 1:num_clusters+1].T
    else:
        k_eigen_matrix = np.flip(eigenvectors, axis=1)[:, :num_clusters].T
    k_eigen_matrix = k_eigen_matrix * np.sign(np.sum(k_eigen_matrix, axis=1).reshape(-1, 1))
    k_eigen_matrix = k_eigen_matrix.T

    if sc_version == 'sym-normed':
        row_norms = np.linalg.norm(k_eigen_matrix, ord=2, axis=1)
        k_eigen_matrix = k_eigen_matrix / row_norms[:, np.newaxis]

    eigenvalues = np.flip(-eigenvalues)

    # Part 4: clustering
    prediction = KMeans(num_clusters, n_init='auto').fit_predict(k_eigen_matrix)

    return prediction, eigenvalues, k_eigen_matrix


def test_sc_round(
        use_data: str,
        num_neighbors: int,
        sc_version: str,
        drop_first_eigen: bool=None,
):
    X, y, num_clusters = get_dataset(use_data)
    timer = time.time()
    prediction, _, _ = spectral_clustering(X, num_clusters, num_neighbors, sc_version, drop_first_eigen)
    acc = compute_clustering_accuracy(y, prediction, num_clusters)
    timer = time.time() - timer
    print(f'{sc_version:>10} --> {use_data:>8}, k={num_neighbors:2}, acc={acc*100:6.2f}%, time elapsed: {timer:.1f} s.')
    return acc
