from sqlalchemy import create_engine
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import mysql.connector
import global_var as gv
import numpy as np


db_connection_str = 'mysql+pymysql://{}@{}/medintegral'.format(gv.USERNAME,gv.SERVER)
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM view_psych_ai', con=db_connection)

df = df.drop(columns=['id','created_at','updated_at'])

for item in df:
    try:
        df[item].replace({"Si": "Yes",
                          "Algunas veces": "Sometimes",
                          "Mas de 1000": "More than 1000",
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

patients_ids = df.patient_id.values
predict_df = df.drop(['patient_id'],axis=1)

predict_df['Gender'] = predict_df['Gender'].str.lower()
predict_df = predict_df.rename(columns={'age':'Age'}, inplace=False)

predict_df['Age'].unique()

predict_df['work_interfere'].replace([np.nan], 'NA', inplace = True)
predict_df['self_employed'].replace([np.nan], 'NA', inplace = True)

# Encoding data
label_encoder = LabelEncoder()
for col in gv.dict_labels:
    if col != 'treatment':
        label_encoder.fit(gv.dict_labels[col])
        predict_df[col] = label_encoder.transform(predict_df[col])

loaded_model = pickle.load(open('/home/eduardo/ai_module/med_integral_ai/psych/trained_model_psych_ai.sav', 'rb'))

y_pred = loaded_model.predict(predict_df)

df_pred = pd.DataFrame(y_pred, columns=["predictions"])

df_pred.insert(0, "patient_id", patients_ids, True)


mydb = mysql.connector.connect(
  host=gv.SERVER,
  user=gv.USERNAME,
  password=gv.PASSWORD,
  database="medintegral"
)

mycursor = mydb.cursor()

for i in range(df_pred.__len__()):
    data = [str(df_pred.predictions[i]),str(df_pred.patient_id[i])]
    sql = "UPDATE patients SET psych_pred = %s, psych_pred_review = 'None' WHERE id = %s"
    mycursor.execute(sql, data)
    mydb.commit()
    print(mycursor.rowcount, "record(s) affected")


