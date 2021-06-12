from sqlalchemy import create_engine
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import mysql.connector


db_connection_str = 'mysql+pymysql://root@localhost/medintegral'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM view_psych_ai', con=db_connection)

df = df.drop(columns=['id','created_at','updated_at'])

for item in df:
    try:
        df[item].replace({"Si": "Yes",
                          "Algunas veces": "Sometimes",
                          "Nunca": "Never",
                          "Frecuentemente": "Often",
                          "Raramente": "Rarely",
                          "No estoy seguro": "Not sure",
                          "No se": "Don't know",
                          "Muy dificil": "Very difficult",
                          "Algo Dificl": "Somewhat difficult",
                          "Algo Facil": "Somewhat easy",
                          "Muy facil": "Very easy",
                          "Algunos de ellos": "Some of them",
                          "A lo mejor": "Maybe"
                          },inplace=True)
    except TypeError:
        continue
        
predict_df = df
patients_ids = df.patient_id.values

predict_df.isnull().sum().max()

defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Creamos una lista con los tipos de datos
intFeatures = ['age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                  'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave',
                  'mentalhealthconsequence',
                  'physhealthconsequence', 'coworkers', 'supervisor', 'mentalhealthinterview',
                  'physhealthinterview',
                  'mentalvsphysical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                  'seek_help']
floatFeatures = []

# Limpiamos de NaNs los datos
for feature in predict_df:
    if feature in intFeatures:
        predict_df[feature] = predict_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        predict_df[feature] = predict_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        predict_df[feature] = predict_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)

# Limpiamos 'Gender'
# Slower case all columm's elements
gender = predict_df['Gender'].str.lower()
# print(gender)

# Select unique elements
gender = predict_df['Gender'].unique()

# Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail",
            "malr",
            "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby",
             "fluid",
             "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman",
             "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)",
              "femail"]

for (row, col) in predict_df.iterrows():

    if str.lower(col.Gender) in male_str:
        predict_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        predict_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        predict_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# Get rid of gibberish
stk_list = ['A little about you', 'p']
predict_df = predict_df[~predict_df['Gender'].isin(stk_list)]

predict_df['age'].fillna(predict_df['age'].median(), inplace=True)

# Fill with media() values < 18 and > 120
s = pd.Series(predict_df['age'])
s[s < 18] = predict_df['age'].median()
predict_df['age'] = s
s = pd.Series(predict_df['age'])
s[s > 120] = predict_df['age'].median()
predict_df['age'] = s

# Ranges of Age
predict_df['age_range'] = pd.cut(predict_df['age'], [0, 20, 30, 65, 100],
                               labels=["0-20", "21-30", "31-65", "66-100"],
                               include_lowest=True)

predict_df['self_employed'] = predict_df['self_employed'].replace([defaultString], 'No')

predict_df['work_interfere'] = predict_df['work_interfere'].replace([defaultString], 'Don\'t know')

# Encoding data
labelDict = {}
for feature in predict_df:
    le = LabelEncoder()
    le.fit(predict_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    predict_df[feature] = le.transform(predict_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

scaler = MinMaxScaler()
predict_df['age'] = scaler.fit_transform(predict_df[['age']])

feature_cols = ['age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave',
                'work_interfere']
X = predict_df[feature_cols]

loaded_model = pickle.load(open('psych/trained_model_psych_ai.sav', 'rb'))

y_pred = loaded_model.predict(X)

df_pred = pd.DataFrame(y_pred, columns=["predictions"])

df_pred.insert(0, "patient_id", patients_ids, True)


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="medintegral"
)

mycursor = mydb.cursor()

for i in range(df_pred.__len__()):
    data = [str(df_pred.predictions[i]),str(df_pred.patient_id[i])]
    sql = "UPDATE patients SET psych_pred = %s, psych_pred_review = 'None' WHERE id = %s"
    mycursor.execute(sql, data)
    mydb.commit()
    print(mycursor.rowcount, "record(s) affected")


