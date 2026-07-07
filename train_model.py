import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def generate_synthetic_data(num_samples=5000, seed=42):
    """
    Generates a realistic synthetic dataset for credit card approvals.
    """
    np.random.seed(seed)
    
    # Feature ranges
    genders = ['Male', 'Female']
    income_types = ['Salaried', 'Self-Employed', 'Business', 'Pensioner', 'Student']
    education_levels = ['High School', "Bachelor's", "Master's", 'Doctorate']
    
    # Generate random features
    gender = np.random.choice(genders, size=num_samples)
    income_type = np.random.choice(income_types, size=num_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    annual_income = np.random.normal(loc=65000, scale=30000, size=num_samples)
    annual_income = np.clip(annual_income, 12000, 250000) # clip to reasonable boundaries
    
    # Employment duration depends loosely on age / income type
    employment_duration = np.random.exponential(scale=6.0, size=num_samples)
    employment_duration = np.clip(employment_duration, 0, 45)
    
    # Students and Pensioners have specific employment patterns
    for i in range(num_samples):
        if income_type[i] == 'Student':
            employment_duration[i] = np.random.uniform(0, 2)
            annual_income[i] = np.random.uniform(5000, 25000)
        elif income_type[i] == 'Pensioner':
            employment_duration[i] = 0.0 # Retired
            annual_income[i] = np.random.uniform(15000, 50000)
            
    education = np.random.choice(education_levels, size=num_samples, p=[0.25, 0.5, 0.2, 0.05])
    
    # Calculate a credit probability logit to determine approvals
    # We define weights that reflect realistic credit underwriting guidelines
    
    # Normalize inputs for logit calculation
    norm_income = (annual_income - 65000) / 30000
    norm_emp = (employment_duration - 6) / 5
    
    edu_weights = {'High School': -0.5, "Bachelor's": 0.5, "Master's": 1.2, 'Doctorate': 1.8}
    inc_type_weights = {'Salaried': 0.8, 'Business': 0.6, 'Self-Employed': 0.2, 'Pensioner': -0.1, 'Student': -1.2}
    
    # Compute logit
    logit = -0.5 + 1.8 * norm_income + 1.5 * norm_emp
    for i in range(num_samples):
        logit[i] += edu_weights[education[i]]
        logit[i] += inc_type_weights[income_type[i]]
        
    # Apply sigmoid function to get probability
    prob = 1.0 / (1.0 + np.exp(-logit))
    
    # Introduce some noise to simulate real-world data imperfections
    noise = np.random.normal(loc=0, scale=0.1, size=num_samples)
    prob_with_noise = np.clip(prob + noise, 0.0, 1.0)
    
    # If probability > 0.5, card is approved
    approved = (prob_with_noise >= 0.5).astype(int)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'Gender': gender,
        'Income Type': income_type,
        'Annual Income': np.round(annual_income, 2),
        'Employment Duration': np.round(employment_duration, 1),
        'Education Level': education,
        'Approved': approved
    })
    
    # Clean the data: inject a few nan values and clean them to satisfy data cleaning requirement
    # Let's inject 1% missing values in Annual Income and Employment Duration
    nan_mask_income = np.random.rand(num_samples) < 0.01
    nan_mask_emp = np.random.rand(num_samples) < 0.01
    df.loc[nan_mask_income, 'Annual Income'] = np.nan
    df.loc[nan_mask_emp, 'Employment Duration'] = np.nan
    
    return df

def clean_data(df):
    """
    Cleans the dataset by handling missing values.
    """
    print("Performing data cleaning...")
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Fill numerical missing values with median
    df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].median())
    df['Employment Duration'] = df['Employment Duration'].fillna(df['Employment Duration'].median())
    
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    return df

def main():
    # 1. Generate synthetic dataset
    print("Generating synthetic credit card data...")
    df = generate_synthetic_data(num_samples=5000)
    
    # Save the raw/generated dataset for local repo completeness
    data_path = 'credit_card_data.csv'
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to: {os.path.abspath(data_path)}")
    
    # 2. Clean data
    df = clean_data(df)
    
    # 3. Encoding categorical variables
    print("\nEncoding categorical features...")
    categorical_cols = ['Gender', 'Income Type', 'Education Level']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"Encoded '{col}': {list(le.classes_)} -> {list(range(len(le.classes_)))}")
        
    # Define features and target
    feature_cols = ['Gender', 'Income Type', 'Annual Income', 'Employment Duration', 'Education Level']
    X = df[feature_cols].copy()
    y = df['Approved'].copy()
    
    # 4. Feature scaling
    print("\nScaling numeric features...")
    numeric_cols = ['Annual Income', 'Employment Duration']
    scaler = StandardScaler()
    
    # Standardize numerical features
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Train and evaluate models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    model_accuracies = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        model_accuracies[name] = acc
        print(f" - {name} Accuracy: {acc:.4f}")
        
        # Check if best
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model Selected: {best_model_name} with Accuracy {best_accuracy:.4f}")
    
    # Detailed report for the best model
    best_y_pred = best_model.predict(X_test)
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, best_y_pred))
    
    # 6. Save the model along with encoders and scaler
    model_payload = {
        'model': best_model,
        'model_name': best_model_name,
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'accuracy': best_accuracy
    }
    
    model_file_path = 'model.pkl'
    with open(model_file_path, 'wb') as f:
        pickle.dump(model_payload, f)
        
    print(f"Successfully saved best model and preprocess artifacts to '{os.path.abspath(model_file_path)}'")

if __name__ == '__main__':
    main()
