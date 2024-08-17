# Heart Disease Prediction using Random Forest Classifier

## Overview

This project uses a Random Forest Classifier to predict the presence of heart disease in patients. The process includes loading the dataset, performing Exploratory Data Analysis (EDA), feature engineering, building the model, and visualizing the results.

## Dataset

The dataset used in this project is the Heart Disease dataset, which contains medical attributes such as age, cholesterol levels, maximum heart rate, and more. The target variable (`target`) indicates whether the patient has heart disease (1) or not (0).

## Project Steps

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame:

```python
file_path = r'C:\Users\Rishab\Downloads\heart-disease (1).csv'
heart_disease_data = pd.read_csv(file_path)

### 2. Exploratory Data Analysis (EDA)

## Various visualizations were used to understand the distribution and relationships of features:

Age Distribution: Visualized with a histogram.
Cholesterol Distribution: Visualized with a histogram.
Max Heart Rate Distribution: Visualized with a histogram.
Age vs Target: Analyzed using a box plot.
Chest Pain Type vs Target: Analyzed using a count plot.
Resting Blood Pressure vs Target: Analyzed using a box plot.
Number of Major Vessels vs Target: Analyzed using a count plot.
