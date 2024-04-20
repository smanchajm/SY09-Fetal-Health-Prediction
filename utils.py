import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist


def plot_Shepard(mds_model, plot=True):
    """Affiche le diagramme de Shepard et retourne un couple contenant les
    dissimilarités originales et les distances apprises par le
    modèle.
    """

    assert isinstance(mds_model, MDS)

    # Inter-distances apprises
    dist = cdist(mds_model.embedding_, mds_model.embedding_)
    idxs = np.tril_indices_from(dist, k=-1)
    dist_mds = dist[idxs]

    # Inter-distances d'origine
    dist = mds_model.dissimilarity_matrix_
    dist_orig = dist[idxs]

    dists = np.column_stack((dist_orig, dist_mds))

    if plot:
        f, ax = plt.subplots()
        range = [dists.min(), dists.max()]
        ax.plot(range, range, 'r--')
        ax.scatter(*dists.T)
        ax.set_xlabel('Dissimilarités')
        ax.set_ylabel('Distances')

    return (*dists.T,)


# Taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    default_kwargs = dict(leaf_font_size=10)
    default_kwargs.update(kwargs or {})

    dendrogram(linkage_matrix, **default_kwargs)


def add_labels(x, y, labels, ax=None):
    """Ajoute les étiquettes `labels` aux endroits définis par `x` et `y`."""

    if ax is None:
        ax = plt.gca()
    for x, y, label in zip(x, y, labels):
        ax.annotate(
            label, [x, y], xytext=(10, -5), textcoords="offset points",
        )

    return ax


# def plot_clustering(data, labels, markers=None, ax=None, **kwargs):
#     """Affiche dans leur premier plan principal les données `data`,
# colorée par `labels` avec éventuellement des symboles `markers`.
#     """

#     if ax is None:
#         ax = plt.gca()

#     # Reduce to two dimensions
#     if data.shape[1] == 2:
#         data_pca = data.to_numpy()
#     else:
#         pca = PCA(n_components=2)
#         data_pca = pca.fit_transform(data)

#     COLORS = np.array(['blue', 'green', 'red', 'purple', 'gray', 'cyan'])
#     _, labels = np.unique(labels, return_inverse=True)
#     colors = COLORS[labels]

#     if markers is None:
#         ax.scatter(*data_pca.T, c=colors)
#     else:
#         MARKERS = "o^sP*+xD"

#         # Use integers
#         markers_uniq, markers = np.unique(markers, return_inverse=True)

#         for marker in range(len(markers_uniq)):
#             data_pca_marker = data_pca[markers == marker, :]
#             colors_marker = colors[markers == marker]
#             ax.scatter(*data_pca_marker.T, c=colors_marker, marker=MARKERS[marker])

#     if 'centers' in kwargs and 'covars' in kwargs:
#         if data.shape[1] == 2:
#             centers_2D = kwargs['centers']
#             covars_2D = kwargs['covars']
#         else:
#             centers_2D = pca.transform(kwargs["centers"])
#             covars_2D = [
#                 pca.components_ @ c @ pca.components_.T
#                 for c in kwargs['covars']
#             ]

#         p = 0.9
#         sig = norm.ppf(p**(1/2))

#         for i, (covar_2D, center_2D) in enumerate(zip(covars_2D, centers_2D)):
#             v, w = linalg.eigh(covar_2D)
#             print(v)
#             v = 2. * sig * np.sqrt(v)

#             u = w[0] / linalg.norm(w[0])
#             if u[0] == 0:
#                 angle = np.pi / 2
#             else:
#                 angle = np.arctan(u[1] / u[0])

#             color = COLORS[i]
#             angle = 180. * angle / np.pi  # convert to degrees
#             ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180. + angle, color=color)
#             ell.set_clip_box(ax.bbox)
#             ell.set_alpha(0.5)
#             ax.add_artist(ell)

#     return ax


