from sqlalchemy import create_engine
import pandas as pd

db_connection_str = 'mysql+pymysql://root@localhost/medintegral'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT ps.* , p.age from '
                 'psych_questionnaires ps left join patients p on ps.patient_id = p.id', con=db_connection)

df = df.drop(columns=['id','created_at','updated_at'])

for item in df:
    try:
        df[item].replace({"Si": "Yes",
                          "Algunas veces": "Sometimes",
                          "Nunca": "Never",
                          "Frecuentemente": "Often",
                          "Raramente": "Rarely",
                          "No estoy seguro": "Don't know",
                          "Muy dificil": "Very difficult",
                          "Algo Dificl": "Somewhat difficult",
                          "Algo Facil": "Somewhat easy",
                          "Muy facil": "Very easy",
                          "Algunos de ellos": "Some of them",
                          "A lo mejor": "Maybe"
                          },inplace=True)
    except TypeError:
        continue

for item in df:
    print(df[item])