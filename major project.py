import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Loading the dataset
file_path = r'C:\Users\Rishab\Downloads\heart-disease (1).csv'  # Using raw string to handle backslashes
heart_disease_data = pd.read_csv(file_path)

print(heart_disease_data.info())

print(heart_disease_data.head())


# EXPLORATORY DATA ANALYSIS (EDA)
# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(heart_disease_data['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Cholesterol Distribution
plt.figure(figsize=(10, 6))
sns.histplot(heart_disease_data['chol'], bins=20, kde=True)
plt.title('Cholesterol Distribution')
plt.show()

# Max Heart Rate Distribution
plt.figure(figsize=(10, 6))
sns.histplot(heart_disease_data['thalach'], bins=20, kde=True)
plt.title('Max Heart Rate Distribution')
plt.show()

# Relationship between age and target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='age', data=heart_disease_data)
plt.title('Age vs Target')
plt.show()

# Chest Pain Type vs Target
plt.figure(figsize=(10, 6))
sns.countplot(x='cp', hue='target', data=heart_disease_data)
plt.title('Chest Pain Type vs Target')
plt.show()

# Resting Blood Pressure vs Target
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='trestbps', data=heart_disease_data)
plt.title('Resting Blood Pressure vs Target')
plt.show()

# Number of Major Vessels vs Target
plt.figure(figsize=(10, 6))
sns.countplot(x='ca', hue='target', data=heart_disease_data)
plt.title('Number of Major Vessels vs Target')
plt.show()

#FEATURE ENGINEERING
# Encode categorical variables
heart_disease_data = pd.get_dummies(heart_disease_data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'], drop_first=True)

# Split the data into features and target
X = heart_disease_data.drop('target', axis=1)
y = heart_disease_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL BUILDING
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# VISUALIZATION
# Feature Importance Visualization
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()



