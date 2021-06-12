
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Validation libraries
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import SVC

train_df = pd.read_csv('../first_csvs/psych.csv')

train_df = train_df.drop(['comments'], axis=1)
train_df = train_df.drop(['state'], axis=1)
train_df = train_df.drop(['Timestamp'], axis=1)

train_df.isnull().sum().max()

defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                  'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                  'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                  'phys_health_interview',
                  'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                  'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)

# clean 'Gender'
# Slower case all columm's elements
gender = train_df['Gender'].str.lower()
# print(gender)

# Select unique elements
gender = train_df['Gender'].unique()

# Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
            "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
             "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman",
             "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]

for (row, col) in train_df.iterrows():

    if str.lower(col.Gender) in male_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# Get rid of bullshit
stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill with media() values < 18 and > 120
s = pd.Series(train_df['Age'])
s[s < 18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s > 120] = train_df['Age'].median()
train_df['Age'] = s

# Ranges of Age
train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"],
                               include_lowest=True)

train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know')

# Encoding data
labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

# Get rid of 'Country'
train_df = train_df.drop(['Country'], axis=1)

scaler = MinMaxScaler()
train_df['Age'] = scaler.fit_transform(train_df[['Age']])

feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = train_df[feature_cols]
y = train_df.treatment

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
pickle.dump(classifier, open(filename, 'wb'))








