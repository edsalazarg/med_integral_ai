USERNAME = "virtual"
PASSWORD = ""
SERVER = "192.168.100.193"

# for item in train_df.columns:
#     print("'{}': {}".format(item,train_df[item].unique()))

dict_labels = {
    'Gender': ['female', 'male','other'],
    'self_employed': ['NA', 'Yes', 'No'],
    'family_history': ['No', 'Yes'],
    'treatment': ['Yes', 'No'],
    'work_interfere': ['Often', 'Rarely', 'Never', 'Sometimes', 'NA'],
    'no_employees': ['6-25', 'More than 1000', '26-100', '100-500', '1-5', '500-1000'],
    'remote_work': ['No', 'Yes'],
    'tech_company': ['Yes', 'No'],
    'benefits': ['Yes', "Don't know", 'No'],
    'care_options': ['Not sure', 'No', 'Yes'],
    'wellness_program': ['No', "Don't know", 'Yes'],
    'seek_help': ['Yes', "Don't know", 'No'],
    'anonymity': ['Yes', "Don't know", 'No'],
    'leave': ['Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult', 'Very easy'],
    'mental_health_consequence': ['No', 'Maybe', 'Yes'],
    'phys_health_consequence': ['No', 'Yes', 'Maybe'],
    'coworkers': ['Some of them', 'No', 'Yes'],
    'supervisor': ['Yes', 'No', 'Some of them'],
    'mental_health_interview': ['No', 'Yes', 'Maybe'],
    'phys_health_interview': ['Maybe', 'No', 'Yes'],
    'mental_vs_physical': ['Yes', "Don't know", 'No'],
    'obs_consequence': ['No', 'Yes']
}