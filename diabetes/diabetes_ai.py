import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import global_var as gv
import os

# Validation libraries
from sklearn.metrics import confusion_matrix, accuracy_score

# Model
from sklearn.svm import SVC

train_df = pd.read_csv('first_csvs/diabetes.csv')

# All this dont need to goes to the other files
train_df.isnull().sum()

train_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = train_df[['Glucose','BloodPressure',
                                                                                  'SkinThickness',
                                                                                  'Insulin','BMI']].replace(0, np.NaN)

for col in train_df.columns:
    train_df.loc[(train_df["Outcome"]==0) & (train_df[col].isnull()),col] = train_df[train_df["Outcome"]==0][col].median()
    train_df.loc[(train_df["Outcome"]==1) & (train_df[col].isnull()),col] = train_df[train_df["Outcome"]==1][col].median()

train_df.isnull().sum()

y = train_df.Outcome
X = train_df.drop(columns=['Outcome'])


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

print('Train:', classifier.score(X_train,y_train))
print('Test:', classifier.score(X_test,y_test))

filename = 'diabetes/trained_model_diabetes_ai.sav'
try:
    os.remove(filename)
except FileNotFoundError:
    print("Previous model file dont exists")

pickle.dump(classifier, open(filename, 'wb'))








