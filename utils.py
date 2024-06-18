import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
import matplotlib as mpl
import seaborn as sns
from sklearn.base import BaseEstimator
import pandas as pd
from scipy.stats import norm
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.utils import check_X_y
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, accuracy_score, classification_report


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




def add_decision_boundary(
    model,
    resolution=100,
    ax=None,
    levels=None,
    label=None,
    color=None,
    region=True,
    model_classes=None,
):
    """Trace une frontière et des régions de décision sur une figure existante.

    :param model: Un modèle scikit-learn ou une fonction `predict`
    :param resolution: La discrétisation en nombre de points par abcisses/ordonnées à utiliser
    :param ax: Les axes sur lesquels dessiner
    :param label: Le nom de la frontière dans la légende
    :param color: La couleur de la frontière
    :param region: Colorer les régions ou pas
    :param model_classes: Les étiquettes des classes dans le cas où `model` est une fonction

    """

    # Set axes
    if ax is None:
        ax = plt.gca()

    # Add decision boundary to legend
    color = "red" if color is None else color
    sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")

    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    def draw_boundaries(XX, YY, Z_num, color):
        # Boundaries
        mask = np.zeros_like(Z_num, dtype=bool)
        for k in range(len(model_classes) - 1):
            mask |= Z_num == k - 1
            Z_num_mask = np.ma.array(Z_num, mask=mask)
            ax.contour(
                XX,
                YY,
                Z_num_mask,
                levels=[k + 0.5],
                linestyles="dashed",
                corner_mask=True,
                colors=[color],
                antialiased=True,
            )

    def get_regions(predict_fun, xy, shape, model_classes):
        Z_pred = predict_fun(xy).reshape(shape)
        cat2num = {cat: num for num, cat in enumerate(model_classes)}
        num2cat = {num: cat for num, cat in enumerate(model_classes)}
        vcat2num = np.vectorize(lambda x: cat2num[x])
        Z_num = vcat2num(Z_pred)
        return Z_num, num2cat

    def draw_regions(ax, model_classes, num2cat, Z_num):
        # Hack to get colors
        # TODO use legend_out = True
        slabels = [str(l) for l in model_classes]
        hdls, hlabels = ax.get_legend_handles_labels()
        hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}

        color_dict = {}
        for label in model_classes:
            if str(label) in hlabels_hdls:
                hdl = hlabels_hdls[str(label)]
                color = hdl.get_markerfacecolor()
                color_dict[label] = color
            else:
                raise Exception("No corresponding label found for ", label)

        colors = [color_dict[num2cat[i]] for i in range(len(model_classes))]
        cmap = mpl.colors.ListedColormap(colors)

        ax.imshow(
            Z_num,
            interpolation="nearest",
            extent=ax.get_xlim() + ax.get_ylim(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.2,
        )

    if isinstance(model, BaseEstimator):
        if model_classes is None:
            model_classes = model.classes_

        if levels is not None:
            if len(model.classes_) != 2:
                raise Exception("Lignes de niveaux supportées avec seulement deux classes")

            # Scikit-learn model, 2 classes + levels
            Z = model.predict_proba(xy)[:, 0].reshape(XX.shape)
            Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)

            # Only 2 classes, simple contour
            ax.contour(
                XX,
                YY,
                Z,
                levels=levels,
                colors=[color]
            )

            draw_regions(ax, model_classes, num2cat, Z_num)
        else:
            # Scikit-learn model + no levels
            Z_num, num2cat = get_regions(model.predict, xy, XX.shape, model_classes)

            draw_boundaries(XX, YY, Z_num, color)
            if region:
                draw_regions(ax, model_classes, num2cat, Z_num)
    else:
        if model_classes is None:
            raise Exception("Il faut spécifier le nom des classes")
        if levels is not None:
            raise Exception("Lignes de niveaux avec fonction non supporté")

        # Model is a predict function, no levels
        Z_num, num2cat = get_regions(model, xy, XX.shape, model_classes)
        draw_boundaries(XX, YY, Z_num, color)
        if region:
            draw_regions(ax, model_classes, num2cat, Z_num)


def scatterplot_pca(
    columns=None, hue=None, style=None, data=None, pc1=1, pc2=2, **kwargs
):
    """Diagramme de dispersion dans le premier plan principal.

    Permet d'afficher un diagramme de dispersion lorsque les données
    ont plus de deux dimensions. L'argument `columns` spécifie la
    liste des colonnes à utiliser pour la PCA dans le jeu de données
    `data`. Les arguments `style` et `hue` permettent de spécifier la
    forme et la couleur des marqueurs. Les arguments `pc1` et `pc2`
    permettent de sélectionner les composantes principales (par défaut
    la première et deuxième). Retourne l'objet `Axes` ainsi que le
    modèle `PCA` utilisé pour réduire la dimension.

    :param columns: Les colonnes quantitatives de `data` à utiliser
    :param hue: La colonne de coloration
    :param style: La colonne du style
    :param data: Le dataFrame Pandas
    :param pc1: La composante en abscisse
    :param pc2: La composante en ordonnée

    """
     # Select relevant columns (should be numeric)
    data_quant = data if columns is None else data[columns]
    data_quant = data_quant.drop(
        columns=[e for e in [hue, style] if e is not None], errors="ignore"
    )

    # Reduce to two dimensions if needed
    if data_quant.shape[1] == 2:
        data_pca = data_quant
        pca = None
    else:
        n_components = max(pc1, pc2)
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_quant)
        data_pca = pd.DataFrame(
            data_pca[:, [pc1 - 1, pc2 - 1]], columns=[f"PC{pc1}", f"PC{pc2}"]
        )

    # Keep name, force categorical data for hue and steal index to
    # avoid unwanted alignment
    if isinstance(hue, pd.Series):
        if not hue.name:
            hue.name = "hue"
        hue_name = hue.name
    elif isinstance(hue, str):
        hue_name = hue
        hue = data[hue]
    elif isinstance(hue, np.ndarray):
        hue = pd.Series(hue, name="class")
        hue_name = "class"

    hue = hue.astype("category")
    hue.index = data_pca.index
    hue.name = hue_name

    if isinstance(style, pd.Series):
        if not style.name:
            style.name = "style"
        style_name = style.name
    elif isinstance(style, str):
        style_name = style
        style = data[style]
    elif isinstance(style, np.ndarray):
        style = pd.Series(style, name="style")
        style_name = "style"

    full_data = data_pca
    if hue is not None:
        full_data = pd.concat((full_data, hue), axis=1)
        kwargs["hue"] = hue_name
    if style is not None:
        full_data = pd.concat((full_data, style), axis=1)
        kwargs["style"] = style_name

    x, y = data_pca.columns
    ax = sns.scatterplot(x=x, y=y, data=full_data, **kwargs)

    return ax, pca


def plot_clustering(data, clus1, clus2=None, ax=None, **kwargs):
    """Affiche les données `data` dans le premier plan principal.

    :param data: Le dataFrame Pandas
    :param clus1: Un premier groupement
    :param clus2: Un deuxième groupement
    :param ax: Les axes sur lesquels dessiner

    """

    if ax is None:
        ax = plt.gca()

    other_kwargs = {e: kwargs.pop(e) for e in ["centers", "covars"] if e in kwargs}

    ax, pca = scatterplot_pca(data=data, hue=clus1, style=clus2, ax=ax, **kwargs)

    if "centers" in other_kwargs and "covars" in other_kwargs:
        # Hack to get colors
        # TODO use legend_out = True
        levels = [str(l) for l in np.unique(clus1)]
        hdls, labels = ax.get_legend_handles_labels()
        colors = [
            artist.get_markerfacecolor()
            for artist, label in zip(hdls, labels)
            if label in levels
        ]
        colors = colors[: len(levels)]

        if data.shape[1] == 2:
            centers_2D = other_kwargs["centers"]
            covars_2D = other_kwargs["covars"]
        else:
            centers_2D = pca.transform(other_kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T for c in other_kwargs["covars"]
            ]

        p = 0.9
        sig = norm.ppf(p ** (1 / 2))

        for covar_2D, center_2D, color in zip(covars_2D, centers_2D, colors):
            v, w = linalg.eigh(covar_2D)
            v = 2.0 * sig * np.sqrt(v)

            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])

            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], angle=180.0 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    return ax, pca


def knn_cross_validation2(X, y, n_folds, n_neighbors_list):
    # Vérifier et valider les entrées X et y
    X, y = check_X_y(X, y)

    # Initialiser le KFold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Générateur pour produire les scores de validation croisée pour chaque n_neighbors
    for n_neighbors in n_neighbors_list:
        # Initialiser le modèle KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        # Calculer les scores de validation croisée
        scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
        # Renvoyer les scores pour la valeur actuelle de n_neighbors
        yield n_neighbors, scores


def select_k_opt(X, y, n_folds, n_list):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {
        'n_neighbors': [],
        'score': []
    }

    best_k = None
    best_score = -np.inf

    for n_neighbors in n_list:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_k = n_neighbors

        for score in scores:
            results['n_neighbors'].append(n_neighbors)
            results['score'].append(score)

    sns.lineplot(data=results, x='n_neighbors', y='score')
    plt.show()
    print(f"Best k: {best_k} with a score of: {best_score}")


def evaluate_model(pipeline, model, X_train, y_train, X_test, y_test, n_folds=10, costs=None):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    X_train_processed = pipeline.fit_transform(X_train, y_train)
    recalls_path = []
    accuracies = []

    for train_index, val_index in kf.split(X_train_processed, y_train):
        X_train_fold, X_val_fold = X_train_processed[train_index], X_train_processed[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)

        if costs is not None:
            # TODO: ajouter le cas où on fait avec des coûts
            pass
        report = classification_report(y_val_fold, y_val_pred, output_dict=True)
        recalls_path.append(report['3.0']['recall'])
        accuracies.append(report['accuracy'])

    X_test_processed = pipeline.transform(X_test)

    model.fit(X_train_processed, y_train)

    y_test_pred = model.predict(X_test_processed)

    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    recall_test = report_test['3.0']['recall']
    accuracy_test = report_test['accuracy']

    return {
        'recalls_path': recalls_path,
        'accuracies': accuracies,
        'recall_path_test': recall_test,
        'accuracy_test': accuracy_test
    }


from collections import defaultdict

def average_classification_reports(classification_reports):
    # Initialisez un dictionnaire pour stocker les moyennes
    average_report = {'1.0': {'precision': 0, 'recall': 0, 'f1-score': 0},
                    '2.0': {'precision': 0, 'recall': 0, 'f1-score': 0},
                    '3.0': {'precision': 0, 'recall': 0, 'f1-score': 0},
                    'accuracy': 0,
                    'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
                    'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0}}

    # Calculez les sommes
    for report in classification_reports:
        for class_value, metrics in report.items():
            if class_value == 'accuracy':
                average_report['accuracy'] += metrics
            elif class_value in ['macro avg', 'weighted avg']:
                average_report[class_value]['precision'] += metrics['precision']
                average_report[class_value]['recall'] += metrics['recall']
                average_report[class_value]['f1-score'] += metrics['f1-score']
            else:
                average_report[class_value]['precision'] += metrics['precision']
                average_report[class_value]['recall'] += metrics['recall']
                average_report[class_value]['f1-score'] += metrics['f1-score']

    # Calculez les moyennes
    num_folds = len(classification_reports)
    for class_value, metrics in average_report.items():
        if class_value == 'accuracy':
            average_report[class_value] /= num_folds
        else:
            average_report[class_value]['precision'] /= num_folds
            average_report[class_value]['recall'] /= num_folds
            average_report[class_value]['f1-score'] /= num_folds

    return average_report


def get_avg_report(pipeline, model, X_train, y_train, X_test, y_test, n_folds=10, costs=None):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    X_train_processed = pipeline.fit_transform(X_train, y_train)
    reports = []

    for train_index, val_index in kf.split(X_train_processed, y_train):
        X_train_fold, X_val_fold = X_train_processed[train_index], X_train_processed[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)

        if costs is not None:
            # TODO: ajouter le cas où on fait avec des coûts
            pass
        report = classification_report(y_val_fold, y_val_pred, output_dict=True)
        reports.append(report)

    average_report = average_classification_reports(reports)
    return average_report
