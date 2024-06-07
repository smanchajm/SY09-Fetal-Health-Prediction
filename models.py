from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif


pipelines = [
    ("Données brutes", Pipeline([
        ("Standardization", FunctionTransformer(validate=True))
    ])),
    ("Données normalisées", Pipeline([
        ("Standardization", StandardScaler())
    ])),
    ("Données brutes avec SV", Pipeline([
        ("Standardization", FunctionTransformer(validate=True)),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ])),
    ("Données normalisées avec SV", Pipeline([
        ("Standardization", StandardScaler()),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]))
]


knn_models = [
    ("Données brutes", Pipeline([
        ("Standardization", FunctionTransformer(validate=True))
    ]), 7),
    ("Données normalisées", Pipeline([
        ("Standardization", StandardScaler())
    ]), 1),
    ("Données brutes avec SV", Pipeline([
        ("Standardization", FunctionTransformer(validate=True)),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]), 3),
    ("Données normalisées avec SV", Pipeline([
        ("Standardization", StandardScaler()),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]), 7)
]


lda_models = pipelines
qda_models = pipelines
nb_models = pipelines
lr_models = pipelines
tree_models = pipelines
