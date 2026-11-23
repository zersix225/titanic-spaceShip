# Task Description
In this competition your task is to predict whether a passenger was transported to an alternate dimension 
during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, 
you're given a set of personal records recovered from the ship's damaged computer system.

## Task target: Transported
This means you must use Transported to map other variables, such as:

```bash
homeplanet --> Transported
age --> Transported
foodcourt --> Transported
```

## Phase 1: Visualize Data

### Step 1: Import Dataset
- Import the dataset from kaggle into your local
- Store the dataset in a cache file path for later use 
### Step 2: Basic Exploration
- Quickly exploration to watched basic info of dataset if you are understand
- Check the number of rows and columns
- Check the data types of each columns
- check for missing values
### Step 3: Clean Dataset
- Reformat name of column
- Find missing values and filled the missing value by statistics calculate
- Check the duplicate records
- Verify that the dataset is ready for analysis
### Step 4:Method analysis
- Univariate
- Bivariate
- Multivariate

## Phase 2: Baseline (archived in `phase2/`)
Archive only. These notebooks still use paths relative to the project root, so they won't run from inside `phase2/` as-is.

| Notebook | What it does |
|---|---|
| `visual.ipynb` | Phase 1 exploration and cleaning, writes `data/clean/train_cleaned.csv` |
| `train_baseline.ipynb` | First CatBoost: features picked by correlation > 0.2, 60:20:20 split |
| `train.ipynb` | CatBoost with engineered features and 5-fold CV |
| `evaluate-model.ipynb` | Confusion matrix, metrics and ROC for the trained model |
| `evaluate-before-after.ipynb` | Compares CatBoost before and after tuning on the same test rows |
| `predict.ipynb` | Refits on all rows and writes the Phase 2 submission (Kaggle 0.80406) |
| `kaggle-setup.ipynb` | Kaggle login and dataset download |

## Experiments (`experiments/`)
Each notebook reads `data/raw/` and writes its CSVs to `submissions/`. They find the project root
themselves, so they run from the project root or from inside `experiments/`.

| File | What it does | Writes |
|---|---|---|
| `0_Model_progress.md` | Write-up of every experiment and its Kaggle score | |
| `1_CatBoost_seeds.ipynb` | CatBoost, seed 42 vs the average of seeds 42, 73 and 101 | `submission_catboost_seeds.csv` |
| `2_LightGBM_seeds.ipynb` | LightGBM, seed 42 vs the three-seed average | `submission_lightgbm_seeds.csv` |
| `3_XGBoost_seeds.ipynb` | XGBoost, seed 42 vs the three-seed average | `submission_xgboost_seeds.csv` |
| `4_Weighted_ensemble.ipynb` | Blends the three models 80% CatBoost, 10% LightGBM, 10% XGBoost | `submission_weighted_80_10_10.csv` |
| `5_Evaluation.ipynb` | Runs notebook 4, then draws ROC curves and AUC for every model and a confusion matrix for the weighted ensemble | (via notebook 4) |

Each notebook installs its libraries in its first cell.

## Submissions (`submissions/`)
| File | Kaggle |
|---|---|
| `submission_weighted_80_10_10.csv` | **0.81084** (best) |
| `submission_catboost_seeds.csv` | 0.81014 |
| `submission_xgboost_seeds.csv` | 0.80547 |
| `submission_lightgbm_seeds.csv` | 0.80523 |

The Phase 2 submission (Kaggle 0.80406) is `phase2/submission_phase2.csv`.

## Folder layout
```
.
├── README.md
├── data/
│   ├── raw/            Kaggle train.csv, test.csv, sample_submission.csv
│   └── clean/          train_cleaned.csv (Phase 1); train_clean.csv / test_clean.csv are left over from
│                       the earlier Phase 3 pipeline and nothing reads them now
├── phase2/             archived Phase 1-2 notebooks
├── experiments/        Phase 3 notebooks, numbered in run order
└── submissions/        every Kaggle submission CSV
```
