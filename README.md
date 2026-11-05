# CISC7204 Assignment 1 — End-to-End Machine Learning Regression Project

A complete machine learning learning project for **CISC7204 (Machine Learning)**, covering the full workflow: **data import → data cleaning → exploratory data analysis (EDA) → model development → cross-validation & hyper-parameter tuning**, applied to multiple regression tasks.

## Research Questions

1. **Car price prediction**: Can we predict the price of a car from its technical specifications (engine size, horsepower, fuel type, body style, etc.)?
2. **Laptop price prediction**: Can we predict the price of a laptop from its hardware attributes (CPU frequency, screen size, RAM, storage, GPU, etc.)?
3. **House price prediction (King County)**: Which features (sqft living area, waterfront, floors, etc.) drive house prices, and can a regression pipeline predict them?
4. **Medical insurance cost prediction**: How do BMI, smoker status, age and other factors affect individual medical charges?

## Datasets & License

All datasets are used for **educational purposes only** in this assignment. None of the data is redistributed as an original work: the CSV files are the course-provided extracts from the IBM Developer Skills Network course *DA0101EN — Data Analysis with Python*, and their upstream origin and license are listed below.

| File(s) | Used in | Upstream source | License |
| --- | --- | --- | --- |
| `auto.csv`, `automobileEDA.csv`, `clean_df.csv`, `usedcars.csv`, `module_5_auto.csv` | Module 01–05 | UCI Machine Learning Repository — **Automobile** data set (derived from `imports-85.data`), DOI [10.24432/C5B01C](https://doi.org/10.24432/C5B01C) — <https://archive.ics.uci.edu/dataset/10/automobile> | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `laptops.csv` (`laptop_pricing_dataset_base.csv` / `_mod1.csv` / `_mod2.csv`) | Module 01–05 | IBM Developer Skills Network course asset, *DA0101EN* — <https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv> (no separate public license is stated by the publisher; course material, coursework use only) | Course-provided asset — all rights with the publisher |
| `kc_house_data_NaN.csv` | Module 06 (Final Project) | **House Sales in King County, USA** — <https://www.kaggle.com/datasets/harlfoxem/housesalesprediction> (distributed in the course as `FinalModule_Coursera/data/kc_house_data_NaN.csv`) | [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| `medical_insurance_dataset.csv` | Module 06 (Practice Project) | **Medical Cost Personal Datasets** (`insurance.csv`, from *Machine Learning with R*, Brett Lantz) — <https://www.kaggle.com/datasets/mirichoi0218/insurance> | Database Contents License (DbCL) as listed on the Kaggle dataset page |

**Attribution and third-party notices**

- The UCI Automobile data set is credited to Jeffrey Schlimmer and the UCI Machine Learning Repository; the CC BY 4.0 terms require attribution and do not imply endorsement of this work by UCI.
- The King County house-sales extract is released under CC0 1.0; attribution is given voluntarily to the original Kaggle uploader.
- The laptop and medical-insurance CSV files are course assets distributed for classroom use; they are reproduced here only as part of submitted coursework.
- This repository does not claim ownership of any dataset. If a rights holder objects to the presence of a file, it will be removed on request.

## Methods & Tools

- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`
- **Data cleaning**: handling `?` / missing values (drop / mean / mode imputation), data type conversion, unit conversion (e.g. MPG → L/100km), normalization, creating dummy variables (`pd.get_dummies`), binning (`pd.cut`)
- **EDA**: correlation analysis & heatmaps, scatter plots, box plots, histograms, pivot tables, Pearson correlation coefficients
- **Modeling**: simple linear regression, multiple linear regression, polynomial regression (degrees 1–11), scikit-learn `Pipeline`, evaluation with R² and prediction accuracy
- **Validation & tuning**: `train_test_split`, k-fold cross-validation, `cross_val_predict`, Ridge regression, `GridSearchCV` (alpha tuning), overfitting analysis

## Key Findings & Limitations

**Key findings**

- Car price shows a strong correlation with engine size and curb weight; highway MPG correlates negatively with price.
- Higher-degree polynomial regression can improve fit on training data but quickly overfits; cross-validation scores reveal the bias–variance trade-off.
- Ridge regression with a tuned `alpha` (via GridSearchCV) stabilizes the polynomial models and improves test-set R².
- For laptops, CPU frequency, RAM and storage positively influence price; for King County houses, sqft living area is the dominant price driver; for insurance, being a smoker has the largest impact on charges.

**Limitations**

- This is a course assignment; models are built for learning the full pipeline rather than production deployment.
- Some datasets contain `?` placeholders that are imputed/dropped; results may differ with other preprocessing choices.
- A few notebooks download datasets from external URLs and therefore require network access to run fully.
- No external/held-out temporal validation beyond the provided train/test splits.

## How to Run

Recommended order (each folder is independent and self-contained):

1. **Module 01 — Data Import**: `Module01_Lab01_Data_Import_Auto.ipynb`, `Module01_Lab02_Data_Import_Laptops.ipynb`
   Reads `auto.csv` / `laptops.csv`, assigns headers, handles `?` → NaN, basic insights.
2. **Module 02 — Data Cleaning**: `Module02_Lab01_Data_Cleaning_Auto.ipynb`, `Module02_Lab02_Data_Cleaning_Laptops.ipynb`
   Missing-value imputation, type conversion, normalization, dummy variables, binning; produces `clean_df.csv`.
3. **Module 03 — EDA**: `Module03_Lab01_EDA_Auto.ipynb`, `Module03_Lab02_EDA_Laptops.ipynb`
   Correlation heatmaps, scatter/box plots, pivot tables. Note: `Module03_Lab02` downloads the laptops dataset from an external URL.
4. **Module 04 — Model Development**: `Module04_Lab01_Model_Development_UsedCars.ipynb`, `Module04_Lab02_Model_Development_Laptops.ipynb`
   Simple / multiple / polynomial regression, pipelines, model comparison and evaluation.
5. **Module 05 — Cross-Validation & Ridge**: `Module05_Lab01_CrossValidation_Ridge_Auto.ipynb`, `Module05_Lab02_CrossValidation_Ridge_Laptops.ipynb`
   Train/test split, cross-validation, overfitting analysis, Ridge + GridSearchCV. Note: `Module05_Lab02` downloads its dataset from an external URL.
6. **Module 06 — Final & Practice Projects**: `Module06_FinalProject_HousePricing.ipynb`, `Module06_PracticeProject_InsuranceCost.ipynb`
   End-to-end pipelines on King County house prices and medical insurance costs.
7. **PreliminaryProject_Pair**: `PreliminaryProject_ProposalVerification.ipynb` + `PreliminaryProject_Proposal.docx`
   Preliminary proposal study (supply–demand matching for a university shuttle service), including research-question verification figures.

### Quick start

```bash
# create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# launch Jupyter and open the notebooks in order
jupyter notebook
```

## Repository Structure

```
├── README.md
├── requirements.txt
├── Module 01/          # data import (auto, laptops)
├── Module 02/          # data cleaning
├── Module 03/          # EDA
├── Module 04/          # model development
├── Module 05/          # cross-validation & ridge regression
├── Module 06/          # final project (house pricing) & practice project (insurance cost)
└── PreliminaryProject_Pair/   # proposal + verification notebook, data & figures
```

## Authors

Course assignment completed as a pair project. Student identifiers have been removed for public sharing.
