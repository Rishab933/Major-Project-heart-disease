Heart Disease Prediction using Random Forest Classifier
Overview
This project involves using a Random Forest Classifier to predict the likelihood of heart disease in patients based on various medical attributes. The steps include data loading, exploratory data analysis (EDA), feature engineering, model building, evaluation, and visualization.

Dataset
The dataset contains various features such as age, cholesterol levels, maximum heart rate, and others, which help in predicting whether a patient has heart disease (target = 1) or not (target = 0).

Project Structure
Data Loading

The dataset is loaded using Pandas.
Exploratory Data Analysis (EDA)

Visualize distributions of important features such as Age, Cholesterol, and Max Heart Rate.
Explore relationships between features such as Age vs Target, Chest Pain Type vs Target, and others.
Feature Engineering

Encode categorical variables.
Split the data into training and testing sets.
Scale numerical features.
Model Building

Build a Random Forest Classifier model using the training data.
Model Evaluation

Evaluate the model’s performance using a confusion matrix and classification report.
Feature Importance Visualization

Visualize the importance of features in predicting heart disease.
Conclusion
This project effectively demonstrates the use of a Random Forest Classifier to predict heart disease. The model's performance is evaluated through various metrics, and the feature importance analysis provides insights into key factors influencing the predictions.
