from sqlalchemy import create_engine
import pandas as pd
import global_var as gv
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import mysql.connector

db_connection_str = 'mysql+pymysql://{}@{}/medintegral'.format(gv.USERNAME,gv.SERVER)
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SHOW DATABASES;', con=db_connection)

print(df)

mydb = mysql.connector.connect(
  host=gv.SERVER,
  user=gv.USERNAME,
  password=gv.PASSWORD,
  database="medintegral"
)

mycursor = mydb.cursor()

sql = "SHOW DATABASES;"
print(mycursor.execute(sql))
