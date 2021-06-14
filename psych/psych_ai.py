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

train_df = pd.read_csv('../first_csvs/psych.csv')

train_df.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace = True)

train_df.drop(train_df[train_df['Age'] < 0].index, inplace = True)
train_df.drop(train_df[train_df['Age'] > 100].index, inplace = True)
train_df['Age'].unique()

train_df.isnull().sum().max()

train_df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'male', inplace = True)

train_df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'female', inplace = True)

train_df["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'other', inplace = True)

train_df['work_interfere'].replace([np.nan], 'NA', inplace = True)
train_df['self_employed'].replace([np.nan], 'NA', inplace = True)

# Get rid of gibberish
stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]

# Encoding data
label_encoder = LabelEncoder()
for col in gv.dict_labels:
    label_encoder.fit(gv.dict_labels[col])
    train_df[col] = label_encoder.transform(train_df[col])

y = train_df.treatment
X = train_df.drop(columns=['treatment'])


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

print('Train:', classifier.score(X_train,y_train))
print('Test:', classifier.score(X_test,y_test))

filename = 'trained_model_psych_ai.sav'
try:
    os.remove(filename)
except FileNotFoundError:
    print("Previous model file dont exists")

pickle.dump(classifier, open(filename, 'wb'))








