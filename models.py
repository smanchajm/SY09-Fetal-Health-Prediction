from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif


pipelines = [
    ("B", Pipeline([
        ("Standardization", FunctionTransformer(validate=True))
    ])),
    ("N", Pipeline([
        ("Standardization", StandardScaler())
    ])),
    ("BSV", Pipeline([
        ("Standardization", FunctionTransformer(validate=True)),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ])),
    ("NSV", Pipeline([
        ("Standardization", StandardScaler()),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]))
]


knn_models = [
    ("B", Pipeline([
        ("Standardization", FunctionTransformer(validate=True))
    ]), 7),
    ("N", Pipeline([
        ("Standardization", StandardScaler())
    ]), 1),
    ("BSV", Pipeline([
        ("Standardization", FunctionTransformer(validate=True)),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]), 3),
    ("NSV", Pipeline([
        ("Standardization", StandardScaler()),
        ("FeatureSelection", SelectKBest(score_func=f_classif, k=6))
    ]), 7)
]


lda_models = pipelines
qda_models = pipelines
nb_models = pipelines
lr_models = pipelines
tree_models = pipelines
