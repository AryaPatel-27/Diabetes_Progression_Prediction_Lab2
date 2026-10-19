# Practical Lab 2: Multivariate Linear Regression, Non-Parametric Models and Cross-Validation

## Project Overview

This project investigates how baseline patient measurements can predict
a numerical measure of diabetes progression one year later.

It uses the Scikit-Learn Diabetes dataset to compare polynomial
regression, decision tree regression, and k-nearest neighbors regression.
A separate Logistic Regression experiment predicts a progression category.

The notebook explains the problem, explores the data, compares models,
and investigates prediction errors.

Although cross-validation appears in the official lab title, the
implemented evaluation uses a fixed train-validation-test split.
K-fold cross-validation is discussed as future work.

## Prediction Tasks

### Regression

Predict the numerical disease progression score one year after baseline.

The patients already have diabetes. The target is not a diabetes diagnosis,
a probability, or a percentage.

### Classification

The separate Logistic Regression experiment predicts:

- Class 0: Progression below the training-set median.
- Class 1: Progression at or above the training-set median.

For this split, the training median is 142.0.

This category definition is a documented educational assumption because
the assignment does not specify classification labels. It is not a
validated clinical risk threshold and remains subject to instructor
confirmation.

## Dataset

The dataset contains:

- 442 patients.
- 10 baseline input features.
- One numerical progression target.

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Coded sex variable |
| bmi | Body mass index |
| bp | Average blood pressure |
| s1 | Total serum cholesterol |
| s2 | Low-density lipoproteins |
| s3 | High-density lipoproteins |
| s4 | Total cholesterol / HDL ratio |
| s5 | Possibly the log of serum triglycerides |
| s6 | Blood sugar level |

Some original serum-feature definitions are uncertain, particularly s5.

The dataset is bundled with Scikit-Learn, so no separate CSV download
is required. The project loads raw features using:

```python
load_diabetes(as_frame=True, scaled=False)
```

Scaling, where required, is fitted using training data through model
pipelines.

## Project Workflow

1. Load the dataset using an object-oriented data loader.
2. Explain the prediction problem and evaluation approach.
3. Explore statistics, missing values, duplicates, distributions,
   scatter plots, and correlations.
4. Explain cleaning decisions.
5. Split patients into training, validation, and test sets.
6. Compare six BMI-only polynomial models.
7. Compare models using all 10 features.
8. Select configurations using validation results.
9. Evaluate selected models on the test set.
10. Inspect large errors and performance across progression groups.
11. Perform a separate Logistic Regression classification analysis.

No missing values or fully duplicated rows were found. All 442
patients were retained.

## Data Split and Evaluation

The split uses random_state=42:

| Set | Patients | Percentage |
|---|---:|---:|
| Training | 331 | 74.89% |
| Validation | 44 | 9.95% |
| Test | 67 | 15.16% |

The same patient splits are used across models.

Training data fits the models. Validation data guides model selection.
Test data provides the final evaluation and is not used to change
model settings.

Regression models are compared using:

- R-squared: Higher is better.
- Mean Absolute Error (MAE): Lower is better.
- Mean Absolute Percentage Error (MAPE): Lower is better.

MAPE is displayed as a percentage.

## Models

| Experiment | Configurations |
|---|---|
| BMI-only polynomial regression | Degrees 0, 1, 2, 3, 4, and 5 |
| Multivariate polynomial regression | Degrees 2 and 3 |
| Decision tree regression | Maximum depths 3 and 5 |
| kNN regression | 5 and 15 neighbors |
| Logistic Regression | C=0.1 and C=1.0 |

The degree-0 model predicts the training target mean.

Polynomial models and kNN use scaling pipelines. Decision trees use
the original feature values.

## Regression Results

Validation MAE selected:

- Degree 5 for BMI-only polynomial regression.
- Degree 2 for multivariate polynomial regression.
- Maximum depth 3 for decision trees.
- Five neighbors for kNN.

The multivariate degree-2 polynomial was selected overall using
validation MAE.

| Selected model | Test R² | Test MAE | Test MAPE |
|---|---:|---:|---:|
| BMI-only polynomial degree 5 | 0.267 | 51.750 | 43.027% |
| Multivariate polynomial degree 2 | 0.399 | 45.493 | 37.949% |
| Decision tree depth 3 | 0.438 | 46.115 | 40.176% |
| kNN, k=5 | 0.471 | 42.209 | 34.653% |

kNN performed best on this test sample. The validation-selected overall
model remains the degree-2 polynomial; test performance was not used
to revise the selection.

## Classification Results

Both Logistic Regression models achieved the same validation F1 and
recall. The stated tie-break rule selected C=0.1.

The progression boundary is 142.0, while the probability decision
threshold is 0.5.

| Test metric | Result |
|---|---:|
| Accuracy | 0.716 |
| Precision | 0.735 |
| Recall | 0.714 |
| F1 | 0.725 |
| ROC AUC | 0.854 |

Test confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 23 | 9 |
| Actual 1 | 10 | 25 |

The classifier identified 25 of 35 class-1 patients and missed 10.

Classification and regression results are kept separate because
they predict different targets and use different metrics.

## Key Findings and Limitations

- Multiple features improved test performance compared with BMI alone.
- The degree-3 multivariate polynomial severely overfit.
- Increasing tree depth did not improve validation performance.
- Errors were often larger for lower or higher progression outcomes.
- Polynomial predictions can fall outside the observed target range.
- Model rankings varied between validation and test samples.
- The small dataset and single split limit confidence in generalization.
- The classification categories are statistical rather than clinical.

This is an educational project. Its results do not establish suitability
for clinical screening or diagnosis.

## Repository Structure

```text
PracticalLab2_DiabetesProgression/
├── README.md
├── requirements.txt
├── .gitignore
├── diabetes_lab.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── data_explorer.py
    ├── data_splitter.py
    ├── polynomial_regression.py
    ├── model_evaluator.py
    ├── model_visualizer.py
    ├── model_inspector.py
    ├── error_analyzer.py
    ├── multivariate_models.py
    ├── regression_visualizer.py
    ├── multivariate_inspector.py
    └── logistic_models.py
```

## Code Organization

The project uses classes to separate responsibilities:

- Data loading and exploration.
- Reproducible splitting.
- Model training and comparison.
- Evaluation and visualization.
- Equation and complexity inspection.
- Prediction-error analysis.

The notebook calls these classes and presents explanations, tables,
plots, and conclusions.

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/PracticalLab2_DiabetesProgression.git
cd PracticalLab2_DiabetesProgression
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it using the command for your terminal.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Open the notebook

Open the repository folder in VS Code and open:

```text
diabetes_lab.ipynb
```

Select the virtual environment as the notebook kernel, then run all
cells from top to bottom.

Alternatively, launch Jupyter:

```bash
jupyter notebook
```

Keep the notebook in the repository root so imports from src work.

## References

- [Scikit-Learn Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [load_diabetes documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Logistic Regression Workshop](https://github.com/ProfEspinosaAIML/LogisticRegressionClassifier_Workshop)
- [kNN Workshop](https://github.com/AlamirConestoga/KNearestNeighbors_Workshop-main)
- [Performance Metrics Workshop](https://github.com/AryaPatel-27/PerformanceMetricsClassification)