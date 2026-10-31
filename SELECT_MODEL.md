# Select a Machine Learning Algorithm (Korn)

**Question:** which algorithm should Phernut train and Fah evaluate?
**Answer:** **CatBoost**, with **LightGBM as a close, much faster alternative.**

## How the comparison was done (same for every model)
- Data: `data/raw/train.csv`, split as Tak did (60/20/20, `random_state=42`, stratified).
- The **test set was not used**. Selection used 5-fold CV on train+val (6,954 rows).
- Folds keep each travel group together (`PassengerId` prefix), so no group leakage.
- Same folds, seed and metrics for all models, **default settings** (tuning is the bonus step).
- Missing values, scaling and one-hot encoding are fitted inside each fold (Pipeline).
- Features: `Deck`, `Side`, `CabinNum` (from Cabin), `GroupSize`, `TotalSpend`, plus the originals.
  `Name`, `PassengerId` and `Cabin` are dropped. People in cryosleep never spend, so their missing spending is set to 0.
- Reproduce: `python select_model.py` (about 1 minute).

## Results
| Model | CV accuracy ± std | F1 | ROC-AUC | Overfit gap (train − CV) | Time / fold | vs best |
|---|---|---|---|---|---|---|
| **CatBoost** | **0.8121 ± 0.0145** | 0.8149 | 0.9029 | 5.0 pts | 16.9 s | best |
| LightGBM | 0.8054 ± 0.0103 | 0.8063 | 0.8976 | 10.4 pts | 0.10 s | close, not significant (p=0.080, wins 4/5 folds) |
| XGBoost | 0.8004 ± 0.0124 | 0.8000 | 0.8929 | 15.6 pts | 0.21 s | lower (p=0.023, 5/5 folds) |
| Random Forest | 0.7993 ± 0.0104 | 0.7928 | 0.8837 | 20.0 pts | 1.7 s | lower (p=0.013, 5/5 folds) |
| Logistic Regression | 0.7734 ± 0.0068 | 0.7777 | 0.8518 | 0.3 pts | 0.03 s | lower (p=0.001, 5/5 folds) |
| Dummy (baseline) | 0.5036 | 0.6699 | 0.5000 | 0.0 pts | 0.01 s | lower |

Chart: `results/model_comparison.png` · Numbers: `results/model_comparison.csv`, `results/fold_accuracy.csv`

## Which one should we choose?
| | Strength | Weakness |
|---|---|---|
| **CatBoost** | Highest accuracy and AUC; smallest gap between train and CV among tree models; reads categories and missing values directly | slower than LightGBM (still only ~50 s for all 5 folds) |
| **LightGBM** | Almost as accurate; very fast, so tuning is cheap | Larger overfit gap; needs one-hot encoding here |
| XGBoost / Random Forest | Reasonable | Clearly below CatBoost, and overfit more |
| Logistic Regression | Simple, no overfitting | About 4 pts lower; cannot capture feature interactions |

**Decision: CatBoost.** It has the best mean on every metric, beats XGBoost, Random Forest and Logistic Regression in all 5 folds, and overfits least among the tree models.

## Honest caveats
- CatBoost vs LightGBM is **not a statistically clear win** (+0.7 pts, p=0.08, only 5 folds). If speed matters for tuning, LightGBM is a fair choice.
- The std across folds (1.0–1.5 pts) is larger than most gaps between the top models, so treat the ranking of the top 3 as a tendency, not a proof.
- Default settings only. Tuning may change the order; that is the Hyper Tuning step.
- Accuracy is a fair metric here because the classes are balanced (50.4% transported).

## Hand-off
- **Phernut (Train):** use CatBoost with `cat_features = [HomePlanet, Destination, Deck, Side, CryoSleep, VIP]`. See `add_features()` in `select_model.py` for the exact features.
- **Fah (Evaluate):** the test split (1,739 rows) is still untouched. Use it once, at the end.
