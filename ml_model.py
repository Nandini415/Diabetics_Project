import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

data=pd.read_csv("Final_Diabetes.csv")
x=data.drop("Outcome",axis=1)
y=data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_model =  make_pipeline(StandardScaler(), RandomForestClassifier(random_state = 18))
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data rf: {accuracy * 100:.2f}%') 
y_pred_full = rf_model.predict(x)
accuracy_full = accuracy_score(y, y_pred_full)
print(f'Accuracy on the entire dataset is rf: {accuracy_full * 100:.2f}%')

knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 4))
knn_pipeline.fit(x_train, y_train)

predictions = knn_pipeline.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on Test Data knn:  {accuracy*100}%")

predictions = knn_pipeline.predict(x)
accuracy = accuracy_score(y, predictions)
print(f"Accuracy on Whole Data knn: {accuracy*100}%")

pickle.dump(rf_model, open("rf_pipeline.pkl", "wb"))