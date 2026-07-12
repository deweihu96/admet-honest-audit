"""v15 -- descriptors + L2 LogisticRegression (standardized), balanced.

RATIONALE: hypothesis-class lever. A regularized linear model is a genuinely
different bias than trees/boosting and a classic strong-simple baseline; on small
assays a well-regularized linear model in a good descriptor space sometimes matches
ensembles. Preprocessing uses QuantileTransformer (rank-based, normal output)
rather than StandardScaler: several RDKit 2D descriptors (e.g. Ipc) have extreme
outliers that destabilize a plain-standardized linear model -- the robust,
defensible choice. class_weight='balanced' handles the minority class. Descriptors
only, since linear models handle the dense physchem space better than sparse bits.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

FEATURES = ["desc"]


def build(seed):
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=seed),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                           random_state=seed))
