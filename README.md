MLOPS/
│
│
├── src/
│   ├── train.py                 # Model training script
│   └── prediction.py            # Loads model and generates predictions
│
├── data/
│   ├── raw/                     # Original transaction dataset
│   └── processed/               # Cleaned and processed data
│
model/
├── preprocessor.pkl
├── Logistic_Regression.pkl
├── Decision_Tree.pkl
└── Random_Forest.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and experiments
│
├── streamlit_app.py             # Streamlit frontend dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files excluded from Git
```



Load Dataset
      ↓
Analyze Target Distribution
      ↓
Sample Data
      ↓
Convert Date & Time
      ↓
EDA & Visualization
      ↓
Train-Test Split
      ↓
Preprocessing
   - Missing Values
   - Scaling
   - Encoding
      ↓
SMOTE Balancing
      ↓
Logistic Regression
      ↓
Decision Tree + GridSearchCV
      ↓
Random Forest
      ↓
Model Evaluation
      ↓
Save Models