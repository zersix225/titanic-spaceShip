"""
Select a Machine Learning algorithm  (Korn)
-------------------------------------------
Goal : compare several algorithms on Spaceship Titanic and pick the best one.

Rules used so the comparison is fair and honest
  * Same split as Tak: 60/20/20, random_state=42, stratified (test is NOT touched here)
  * Selection uses 5-fold CV on train+val only (80% of the data)
  * Folds keep each travel group together (PassengerId prefix) -> no group leakage
  * Same folds, seed and metrics for every model; default settings (no tuning yet)
  * Missing-value filling / scaling / encoding is fitted INSIDE each fold (Pipeline)

Run:   python select_model.py
Out :  results/model_comparison.csv   results/model_comparison.png
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA = Path("data/raw/train.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]
NUM = ["Age", "CabinNum", "GroupSize", "TotalSpend"] + SPEND


class CatBoostCV(ClassifierMixin, BaseEstimator):
    """Thin wrapper so CatBoost (with cat_features) works inside sklearn cross_validate."""

    def __init__(self, seed=SEED):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = CatBoostClassifier(cat_features=CAT, random_seed=self.seed, verbose=0,
                                         allow_writing_files=False, thread_count=-1)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X).astype(int).ravel()

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ---------------------------------------------------------------- 1. data + split
df = pd.read_csv(DATA)
y = df["Transported"].astype(int)

# Same two-step split as Tak (indices only, so any column set gives the same rows)
idx_tv, idx_test = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=SEED, stratify=y.loc[idx_tv])
print(f"split  train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)} (test not used)")

dev = df.loc[idx_tv].copy()          # train + val = data used for model selection
y_dev = y.loc[idx_tv]


# ---------------------------------------------------------------- 2. row-wise features
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Only row-wise rules and counts of IDs; nothing learned from the target."""
    d = d.copy()
    # Rule: people in cryosleep never spend (0 of 3,037 in train do) -> missing spend = 0
    asleep = d["CryoSleep"] == True  # noqa: E712
    for c in SPEND:
        d.loc[asleep & d[c].isna(), c] = 0.0
    cabin = d["Cabin"].str.split("/", expand=True)
    d["Deck"], d["CabinNum"], d["Side"] = cabin[0], pd.to_numeric(cabin[1]), cabin[2]
    group = d["PassengerId"].str[:4]
    d["GroupSize"] = group.map(group.value_counts())
    d["TotalSpend"] = d[SPEND].sum(axis=1, min_count=1)
    for c in ["CryoSleep", "VIP"]:           # booleans -> text so NaN becomes its own category
        d[c] = d[c].map({True: "True", False: "False"})
    for c in CAT:
        d[c] = d[c].fillna("Missing").astype(str)
    return d[CAT + NUM]


X_dev = add_features(dev)
groups = dev["PassengerId"].str[:4].values
print("features:", X_dev.shape[1], "| missing numeric cells left:", int(X_dev[NUM].isna().sum().sum()))


# ---------------------------------------------------------------- 3. preprocessors
def log_spend(a):
    return np.log1p(np.clip(a, 0, None))


def linear_pre():       # Logistic Regression: impute + log + scale + one-hot
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("log", FunctionTransformer(log_spend, feature_names_out="one-to-one")),
                    ("sc", StandardScaler())])
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def forest_pre():       # Random Forest: needs no NaN -> impute + one-hot
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def boost_pre():        # LightGBM / XGBoost take NaN directly -> only one-hot
    return ColumnTransformer([("num", "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


models = {
    "Dummy (baseline)":    Pipeline([("pre", boost_pre()), ("m", DummyClassifier(strategy="prior"))]),
    "Logistic Regression": Pipeline([("pre", linear_pre()), ("m", LogisticRegression(max_iter=2000))]),
    "Random Forest":       Pipeline([("pre", forest_pre()), ("m", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
    "XGBoost":             Pipeline([("pre", boost_pre()), ("m", XGBClassifier(random_state=SEED, n_jobs=-1, verbosity=0))]),
    "LightGBM":            Pipeline([("pre", boost_pre()), ("m", LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1))]),
    # CatBoost reads the text categories and NaN itself, so it gets the raw feature table
    "CatBoost":            CatBoostCV(),
}

# ---------------------------------------------------------------- 4. same CV for all
cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(X_dev, y_dev, groups))
scoring = {"acc": "accuracy", "f1": "f1", "auc": "roc_auc"}

rows, fold_acc = [], {}
for name, model in models.items():
    t0 = time.time()
    r = cross_validate(model, X_dev, y_dev, cv=folds, scoring=scoring,
                       return_train_score=True, n_jobs=1)
    fold_acc[name] = r["test_acc"]
    rows.append({
        "model": name,
        "cv_accuracy": r["test_acc"].mean(),
        "std": r["test_acc"].std(),
        "f1": r["test_f1"].mean(),
        "roc_auc": r["test_auc"].mean(),
        "train_accuracy": r["train_acc"].mean(),
        "overfit_gap_pts": (r["train_acc"].mean() - r["test_acc"].mean()) * 100,
        "fit_sec_per_fold": r["fit_time"].mean(),
    })
    print(f"{name:20s} acc={r['test_acc'].mean():.4f} ±{r['test_acc'].std():.4f}  ({time.time()-t0:.0f}s)")

res = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)

# Is each model clearly worse than the best, or only close?  (paired t-test, same folds)
best = res.loc[0, "model"]
verdict, wins = [], []
for _, r in res.iterrows():
    if r["model"] == best:
        verdict.append("best"); wins.append("-")
        continue
    diff = fold_acc[best] - fold_acc[r["model"]]
    p = stats.ttest_rel(fold_acc[best], fold_acc[r["model"]]).pvalue
    wins.append(f"{int((diff > 0).sum())}/{N_FOLDS}")
    verdict.append(f"clearly lower (p={p:.3f})" if p < 0.05 else f"close, not significant (p={p:.3f})")
res["best_wins_folds"] = wins
res["vs_best"] = verdict

res.round(4).to_csv(OUT / "model_comparison.csv", index=False)
pd.DataFrame(fold_acc).round(4).to_csv(OUT / "fold_accuracy.csv", index_label="fold")

# ---------------------------------------------------------------- 5. chart
fig, ax = plt.subplots(figsize=(8, 4))
order = res.iloc[::-1]
colors = ["#2b78d4" if m == best else "#9aa3b2" for m in order["model"]]
ax.barh(order["model"], order["cv_accuracy"], xerr=order["std"], color=colors, capsize=3)
for i, (a, s) in enumerate(zip(order["cv_accuracy"], order["std"])):
    ax.text(a + s + 0.004, i, f"{a:.4f}", va="center", fontsize=9)
ax.set_xlim(0.45, 0.86)
ax.set_xlabel(f"Mean CV accuracy (±1 std, {N_FOLDS} grouped folds, default settings)")
ax.set_title("Model selection: which algorithm predicts best?")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150)

print("\n", res.round(4).to_string(index=False))
