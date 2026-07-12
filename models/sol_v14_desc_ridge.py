"""sol_v14 -- descriptors + Ridge (quantile-normalized) regression.

RATIONALE: hypothesis-class lever. A regularized linear model is a genuinely
different bias than trees/boosting and a classic solubility baseline (logS is
famously near-linear in a few physicochemical descriptors, e.g. the ESOL model).
Preprocessing uses QuantileTransformer (rank-based, normal output) rather than
StandardScaler: several RDKit 2D descriptors (e.g. Ipc) have extreme outliers that
make a plain-standardized linear model numerically unstable -- the robust,
defensible choice for a linear model on raw descriptors. Descriptors only.
"""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

FEATURES = ["desc"]


def build(seed):
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=seed),
        Ridge(alpha=1.0, random_state=seed))
