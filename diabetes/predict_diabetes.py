from sqlalchemy import create_engine
import pandas as pd
import pickle
import mysql.connector
import global_var as gv


db_connection_str = 'mysql+pymysql://{}@{}/medintegral'.format(gv.USERNAME,gv.SERVER)
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM view_diabetes_ai', con=db_connection)

df = df.drop(columns=['created_at'])

patients_ids = df.patient_id.values
predict_df = df.drop(['patient_id'],axis=1)

loaded_model = pickle.load(open('/home/eduardo/ai_module/med_integral_ai/diabetes/trained_model_diabetes_ai.sav', 'rb'))

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
    sql = "UPDATE patients SET diabetes_pred = %s, diabetes_pred_review = 'None' WHERE id = %s"
    mycursor.execute(sql, data)
    mydb.commit()
    print(mycursor.rowcount, "record(s) affected")


