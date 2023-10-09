import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


data = pd.read_csv("stroke_dataset.csv")
data = data.dropna()
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
          'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
X_copy = X.copy()  
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'ever_married', 'Residence_type', 'smoking_status']



X_copy['gender'] = label_encoder.fit_transform(X_copy['gender'])
X_copy['smoking_status'] = label_encoder.fit_transform(X_copy['smoking_status'])
X_copy['Residence_type'] = label_encoder.fit_transform(X_copy['Residence_type'])
X_copy['ever_married'] = label_encoder.fit_transform(X_copy['ever_married'])


y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for col in categorical_columns:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

#SMOTE cho oversampling và RandomUnderSampler cho undersampling
oversample = SMOTE(sampling_strategy=0.5, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

#  oversampling và undersampling 
resampling_pipeline = Pipeline([
    ('oversample', oversample),
    ('undersample', undersample)
])
X_train_resampled, y_train_resampled = resampling_pipeline.fit_resample(X_train, y_train)
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
for model_name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print("\n")




