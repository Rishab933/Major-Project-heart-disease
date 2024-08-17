# Heart Disease Prediction Using Random Forest Classifier

## Overview

This project uses a Random Forest Classifier to predict the likelihood of heart disease in patients based on various medical attributes. The workflow includes data loading, exploratory data analysis (EDA), feature engineering, model building, evaluation, and visualization.

## Dataset

The dataset used in this project contains various medical attributes, such as age, cholesterol levels, and maximum heart rate, to predict whether a patient has heart disease (target = 1) or not (target = 0).

## Project Structure

### 1. Data Loading

The dataset is loaded from a CSV file using Pandas.

### 2. Exploratory Data Analysis (EDA)

The following visualizations are created:
- Age Distribution
- Cholesterol Distribution
- Max Heart Rate Distribution
- Relationship between Age and Target
- Chest Pain Type vs Target
- Resting Blood Pressure vs Target
- Number of Major Vessels vs Target

### 3. Feature Engineering

- **Categorical Encoding**: Convert categorical variables to dummy variables.
- **Data Splitting**: Split the data into training and testing sets.
- **Feature Scaling**: Standardize numerical features.

### 4. Model Building

- Build and train a Random Forest Classifier using the training data.

### 5. Model Evaluation

- Evaluate the model’s performance using:
  - Confusion Matrix
  - Classification Report
