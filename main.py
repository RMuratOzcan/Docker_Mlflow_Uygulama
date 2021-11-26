#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Data analysis library
import numpy as np
import pandas as pd

# Machine Learning library
import sklearn
from sklearn.metrics import roc_curve, auc, accuracy_score, plot_confusion_matrix, plot_roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Model experimentation library
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Plotting library
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# In[7]:


data = pd.read_csv(r"C:\Users\User\Desktop\Placement_Data_Full_Class.csv")

# In[8]:


data.head()

# In[14]:


experiment_name = "campus_recruitment_experiments_v1"
artifact_repository = './mlflow-run'

# Provide uri and connect to your tracking server
mlflow.set_tracking_uri("http://localhost:5000/")

# Initialize MLflow client
client = MlflowClient()

experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_repository)

# In[28]:


exclude_feature = ['sl_no', 'salary', 'status']
# Define Target columns

# Define numeric and categorical features
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numeric_features = [col for col in numeric_columns if col not in exclude_feature]
categorical_features = [col for col in categorical_columns if col not in exclude_feature]

# Define final feature list for training and validation
features = numeric_features + categorical_features
# Final data for training and validation
data = data[features]
data = data.fillna(0)

# Split data in train and vlaidation
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.15, random_state=10)

# Perform label encoding for categorical variable
for feature in categorical_features:
    le = LabelEncoder()
    le.fit(X_train.loc[:, feature])
    X_train.loc[:, feature] = le.transform(X_train.loc[:, feature])
    X_valid.loc[:, feature] = le.transform(X_valid.loc[:, feature])


# In[29]:


def model_experimentation(classifier, param, model_name, run_name):
    # Launching Multiple Runs in One Program.This is easy to do because the ActiveRun object returned by mlflow.start_run() is a
    # Python context manager. You can “scope” each run to just one block of code as follows:
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Get run id
        run_id = run.info.run_uuid

        # Provide brief notes about the run
        MlflowClient().set_tag(run_id,
                               "mlflow.note.content",
                               "This is experiment for exploring different machine learning models for Campus Recruitment Dataset")

        # To enable autologging for scikit-learn estimators.
        # 1) Log estimator parameters
        # 2) Log common metrics for classifier
        # 3) Log model Artifacts
        mlflow.sklearn.autolog()

        # Define custom tag
        tags = {"Application": "Payment Monitoring Platform",
                "release.candidate": "PMP",
                "release.version": "2.2.0"}

        # Set Tag
        mlflow.set_tags(tags)

        # Log python environment details

        # Perform model training
        clf = classifier(**param)
        clf.fit(X_train, y_train)

        # Perform model evaluation
        valid_prediction = clf.predict_proba(X_valid)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_valid, valid_prediction)
        roc_auc = auc(fpr, tpr)  # compute area under the curve
        print("=====================================")
        print("Validation AUC:{}".format(roc_auc))
        print("=====================================")

        # log metrics
        mlflow.log_metrics({"Validation_AUC": roc_auc})


# In[30]:


classifier = LogisticRegression
param = {"C": 1, "random_state": 20}
model_name = 'Lt'
run_name = 'LogisticRegression_model'
model_experimentation(classifier, param, model_name, run_name)

mlflow.register_model("http://localhost:5000/#/models,
                      "lojistik_regresyon")
# In[ ]:




