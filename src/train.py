from pathlib import Path
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_PATH = Path("./data/SAML-D.csv")
MODEL_DIR = Path("./model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# LOAD DATA


df = pd.read_csv(DATA_PATH, header=0, sep=",")

print("Dataset shape:", df.shape)
print(df.head())

count_laundering = (df["Is_laundering"] == 1).sum()
print("Number of rows where Is_laundering = 1:", count_laundering)

# DOWNSAMPLING

laundering_df = df[df["Is_laundering"] == 1]
normal_df = df[df["Is_laundering"] == 0]

target_rows = 500_000
normal_sample_size = target_rows - len(laundering_df)

if normal_sample_size < 0 or normal_sample_size > len(normal_df):
    raise ValueError(
        "Cannot create requested sample size with the available class counts."
    )

normal_sample_df = normal_df.sample(
    n=normal_sample_size,
    random_state=42
)

df_sample = pd.concat(
    [laundering_df, normal_sample_df],
    ignore_index=True
)

df_sample = (
    df_sample
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print("Original dataset shape:", df.shape)
print("Sample dataset shape:", df_sample.shape)

print(
    "Target distribution:\n",
    df_sample["Is_laundering"].value_counts()
)

print(
    "All laundering rows retained:",
    (df_sample["Is_laundering"] == 1).sum()
    ==
    (df["Is_laundering"] == 1).sum()
)

# DATE/TIME CONVERSION

df_sample["Date"] = (
    pd.to_datetime(
        df_sample["Date"],
        errors="coerce"
    )
    .dt.strftime("%Y%m%d")
    .pipe(pd.to_numeric, errors="coerce")
    .astype("Int64")
)

df_sample["Time"] = (
    pd.to_datetime(
        df_sample["Time"],
        format="%H:%M:%S",
        errors="coerce"
    )
    .dt.strftime("%H%M%S")
    .pipe(pd.to_numeric, errors="coerce")
    .astype("Int64")
)

print(df_sample[["Date", "Time"]].head())
print(df_sample.dtypes)

# EDA

sns.countplot(
    x="Is_laundering",
    data=df_sample
)

plt.title("Laundering Target Distribution")
plt.tight_layout()
plt.show()

payment_counts = pd.crosstab(
    df_sample["Payment_type"],
    df_sample["Is_laundering"]
)

print(payment_counts)

payment_counts.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Payment Type vs Laundering Status")
plt.xlabel("Payment Type")
plt.ylabel("Number of Transactions")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    ["Normal (0)", "Laundering (1)"]
)

plt.tight_layout()
plt.show()

# SPLIT FEATURES / TARGET

X = df_sample.drop(columns=["Is_laundering"])
Y = df_sample["Is_laundering"].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

# FEATURE TYPES

numerical_features = (
    X.select_dtypes(include=["number"])
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(
        include=["object", "category", "string", "bool"]
    )
    .columns
    .tolist()
)

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# PREPROCESSING


numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


# SMOTE


smote = SMOTE(random_state=42)

X_train_smote, Y_train_smote = smote.fit_resample(
    X_train_processed,
    Y_train
)

print("After SMOTE:", X_train_smote.shape)


# LOGISTIC REGRESSION


logistic_model = LogisticRegression(
    solver="lbfgs",
    max_iter=10000,
    C=1,
    penalty="l2"
)

logistic_model.fit(
    X_train_smote,
    Y_train_smote
)

y_pred_Logistic = logistic_model.predict(
    X_test_processed
)

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_Logistic))

print("\nLogistic Regression Classification Report:")
print(classification_report(Y_test, y_pred_Logistic))

joblib.dump(
    logistic_model,
    MODEL_DIR / "Logistic_Regression.pkl"
)


# DECISION TREE


dt_model = DecisionTreeClassifier(
    random_state=42
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "ccp_alpha": [0.0, 0.001, 0.01]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search_dt = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search_dt.fit(
    X_train_smote,
    Y_train_smote
)

print(
    "Decision Tree best parameters:",
    grid_search_dt.best_params_
)

print(
    "Decision Tree best CV score:",
    grid_search_dt.best_score_
)

best_dt_model = grid_search_dt.best_estimator_

y_pred_DT = best_dt_model.predict(
    X_test_processed
)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_DT))

print("\nDecision Tree Classification Report:")
print(classification_report(Y_test, y_pred_DT))

joblib.dump(
    best_dt_model,
    MODEL_DIR / "Decision_Tree.pkl"
)


# RANDOM FOREST


rf_model = RandomForestClassifier(
    bootstrap=True,
    n_estimators=50,
    max_depth=15,
    random_state=2,
    max_features=10,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    Y_train_smote
)

y_pred_RF = rf_model.predict(
    X_test_processed
)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred_RF))

print("\nRandom Forest Classification Report:")
print(classification_report(Y_test, y_pred_RF))

joblib.dump(
    rf_model,
    MODEL_DIR / "Random_Forest.pkl"
)

print(
    "\nTraining workflow completed."
)

print(
    "Models saved in:",
    MODEL_DIR.resolve()
)