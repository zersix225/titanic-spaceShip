# Spaceship Titanic model progress

The weighted ensemble reached **0.81084 on Kaggle**, up from the reported **0.804** baseline. That is about **0.68 percentage points** of accuracy and clears the 0.81 target. Most of the gain came when we changed the feature set and training recipe together. A controlled comparison then showed a further gain from averaging CatBoost seeds.

## Notebooks in this folder

Each notebook finds the project root by itself, so it runs from the project root or from inside `experiments/`. Every notebook reads `data/raw/` and writes its submission to `submissions/`.

| Notebook | What it does | Writes |
|---|---|---|
| [1_CatBoost_seeds.ipynb](1_CatBoost_seeds.ipynb) | CatBoost, seed 42 vs the average of seeds 42, 73 and 101 | `submission_catboost_seeds.csv` |
| [2_LightGBM_seeds.ipynb](2_LightGBM_seeds.ipynb) | LightGBM, seed 42 vs the three-seed average | `submission_lightgbm_seeds.csv` |
| [3_XGBoost_seeds.ipynb](3_XGBoost_seeds.ipynb) | XGBoost, seed 42 vs the three-seed average | `submission_xgboost_seeds.csv` |
| [4_Weighted_ensemble.ipynb](4_Weighted_ensemble.ipynb) | Trains all three models and blends them 80/10/10. This is the best submission | `submission_weighted_80_10_10.csv` |
| [5_Evaluation.ipynb](5_Evaluation.ipynb) | Runs notebook 4, then draws the out-of-fold ROC curve and AUC for every model and a confusion matrix for the weighted ensemble | `submission_weighted_80_10_10.csv` (via notebook 4) |

All four use the same features, the same five stratified folds (`random_state=42`), seeds 42, 73 and 101, and a 0.5 threshold.

## How the winning model changed

1. **Starting point · 0.804.** The original CatBoost pipeline already used group size, solo travel, cabin deck and side, cryosleep-based spending rules, and log spending. It selected the tree count by cross-validation, then fitted one model on all labeled passengers. [Original training notebook](../phase2/train.ipynb) · [Original prediction notebook](../phase2/predict.ipynb)
2. **Expanded-feature CatBoost · 0.80967.** We used the unlabeled passenger manifest to count cabin occupants and surname occurrences, calculate mean spending within each travel group, and fill missing cabin fields or home planet when known values agreed within the group. We added age, spending, cabin-number and passenger-group bands. CatBoost trained at depth 6 for 2,000 iterations with a learning rate of 0.01. We averaged test probabilities from five stratified folds. These changes were tested together, so the roughly 0.00567 gain cannot be attributed to any one feature.
3. **Three-seed CatBoost · 0.81014.** We kept the features, folds and parameters fixed and trained with seeds 42, 73 and 101. We averaged probabilities from all 15 fits and applied a 0.5 threshold. Seed 42 reproduced every prediction of the 0.80967 submission. Averaging changed 28 of 4,277 test labels, raised out-of-fold accuracy from 0.81261 to 0.81318, and added **0.00047 on Kaggle**. [Three-seed CatBoost notebook](1_CatBoost_seeds.ipynb) · [Three-seed CatBoost submission](../submissions/submission_catboost_seeds.csv)
4. **Weighted ensemble · 0.81084.** We combined the three seed-averaged model probabilities using 80% CatBoost, 10% LightGBM and 10% XGBoost, then applied a 0.5 threshold. This added **0.00070 on Kaggle** over three-seed CatBoost. [Weighted ensemble notebook](4_Weighted_ensemble.ipynb)

## Experiment results

Kaggle scores below are the reported leaderboard results. Local scores use out-of-fold predictions on labeled passengers.

| Experiment | Kaggle accuracy | What we learned |
|---|---:|---|
| Equal-weight CatBoost, LightGBM and XGBoost | 0.80687 | The blend narrowly beat seed-42 CatBoost locally, but seed-42 CatBoost scored higher on Kaggle. |
| CatBoost with deck, side and cabin-number block combined | 0.80874 | With passenger groups held together for validation, accuracy rose from 0.81353 to 0.81399. Only one of five folds improved. This experiment also used grouped folds to train its submission, so the leaderboard difference does not isolate the cabin feature. |
| [Three-seed LightGBM](2_LightGBM_seeds.ipynb) | 0.80523 | Averaging seeds gave 0.81169 locally, up from 0.81111 for seed 42 alone. A rerun on 2026-11-21 reproduced both numbers. |
| [Three-seed XGBoost](3_XGBoost_seeds.ipynb) | 0.80547 | Averaging seeds gave 0.81054 locally. XGBoost edged LightGBM on Kaggle despite their reversed local ranking. |
| [80% CatBoost, 10% LightGBM, 10% XGBoost](4_Weighted_ensemble.ipynb) | **0.81084** | The weighted blend changed 33 test labels and scored 0.81261 locally, below three-seed CatBoost at 0.81318 on the same folds. |

Two earlier experiments never got a Kaggle score:

- **Histogram gradient boosting candidate:** 0.81169 locally. It was a standalone script with fewer features and one seed. The script has since been removed from the folder.
- **Equal-weight blend of three-seed LightGBM and XGBoost:** 0.81123 locally. It was dropped when LightGBM and XGBoost were split into their own notebooks.

## Reproduce the best submission

Open [4_Weighted_ensemble.ipynb](4_Weighted_ensemble.ipynb) and choose **Run All Cells**. It works from either the project root or `experiments/`. It reads the CSVs in `data/raw/` and writes `submissions/submission_weighted_80_10_10.csv`. If `submissions/submission_catboost_seeds.csv` exists, it also prints how many predictions differ from the three-seed CatBoost submission.
