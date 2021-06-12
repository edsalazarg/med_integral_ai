# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:30:29 2020

@author: edsal
"""

import numpy as np
import pandas as pd
import random as rd
from sqlalchemy import create_engine
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import pickle
import global_var as gv

class Psych_AI:
    def __init__(self, dataframe, population=20, prob_crsvr=1, prob_mutation=0.3, generations=1000):
        self.dataframe = dataframe
        self.x, self.y = self.parsing_df()
        self.population = population
        self.prob_crsvr = prob_crsvr
        self.prob_mutation = prob_mutation
        self.generations = generations
        self.start_optimization()

    def parsing_df(self):
        # leemos los datos
        train_df = self.dataframe

        # Revolvemos los datos
        train_df = train_df.drop(['comments'], axis=1)
        train_df = train_df.drop(['state'], axis=1)
        train_df = train_df.drop(['Timestamp'], axis=1)

        train_df.isnull().sum().max()

        defaultInt = 0
        defaultString = 'NaN'
        defaultFloat = 0.0

        # Creamos una lista con los tipos de datos
        intFeatures = ['Age']
        stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                          'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave',
                          'mental_health_consequence',
                          'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                          'phys_health_interview',
                          'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                          'seek_help']
        floatFeatures = []

        # Limpiamos de NaNs los datos
        for feature in train_df:
            if feature in intFeatures:
                train_df[feature] = train_df[feature].fillna(defaultInt)
            elif feature in stringFeatures:
                train_df[feature] = train_df[feature].fillna(defaultString)
            elif feature in floatFeatures:
                train_df[feature] = train_df[feature].fillna(defaultFloat)
            else:
                print('Error: Feature %s not recognized.' % feature)

        # Limpiamos 'Gender'
        # Slower case all columm's elements
        gender = train_df['Gender'].str.lower()
        # print(gender)

        # Select unique elements
        gender = train_df['Gender'].unique()

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

        for (row, col) in train_df.iterrows():

            if str.lower(col.Gender) in male_str:
                train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

            if str.lower(col.Gender) in female_str:
                train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

            if str.lower(col.Gender) in trans_str:
                train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

        # Get rid of gibberish
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
        train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100],
                                       labels=["0-20", "21-30", "31-65", "66-100"],
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

        feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave',
                        'work_interfere']
        return train_df[feature_cols], train_df.treatment

    def objective_value(self,x, y, chromosome):
        lb_x, ub_x = .1, 10
        len_x = (len(chromosome) // 2)

        precision_x = (ub_x - lb_x) / ((2 ** len_x) - 1)

        z = 0
        t = 1
        x_bit_sum = 0
        for i in range(len(chromosome) // 2):
            x_bit = chromosome[-t] * (2 ** z)
            x_bit_sum += x_bit
            t += 1
            z += 1

        c_hyperparameter = (x_bit_sum * precision_x) + lb_x

        # USING train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        model = svm.SVC(kernel="rbf", C=c_hyperparameter)
        model.fit(x_train, np.ravel(y_train))

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        return c_hyperparameter, accuracy

    def find_parents_ts(self, all_solutions, x, y):
        parents = np.empty((0, np.size(all_solutions, 1)))
        for i in range(2):
            indices_list = np.random.choice(len(all_solutions), 3, replace=False)
            # Selecting 3 random solutions
            posb_parent_1 = all_solutions[indices_list[0]]
            posb_parent_2 = all_solutions[indices_list[1]]
            posb_parent_3 = all_solutions[indices_list[2]]

            # Testing best one
            obj_func_parent_1 = self.objective_value(x, y, posb_parent_1)[1]
            obj_func_parent_2 = self.objective_value(x, y, posb_parent_2)[1]
            obj_func_parent_3 = self.objective_value(x, y, posb_parent_3)[1]

            min_obj_func = max(obj_func_parent_1, obj_func_parent_2,
                               obj_func_parent_3)

            if min_obj_func == obj_func_parent_1:
                selected_parent = posb_parent_1
            elif min_obj_func == obj_func_parent_2:
                selected_parent = posb_parent_2
            else:
                selected_parent = posb_parent_3

            parents = np.vstack((parents, selected_parent))

        parent_1 = parents[0, :]
        parent_2 = parents[1, :]

        return parent_1, parent_2

    def crossover(self, parent_1, parent_2, prob_crsvr=1):
        child_1 = np.empty((0, len(parent_1)))
        child_2 = np.empty((0, len(parent_2)))

        rand_num_to_crsvr_or_not = np.random.rand()

        if rand_num_to_crsvr_or_not < prob_crsvr:
            index_1 = np.random.randint(0, len(parent_1))
            index_2 = np.random.randint(0, len(parent_1))

            while index_1 == index_2:
                index_2 = np.random.randint(0, len(parent_1))

            index_parent_1 = min(index_1, index_2)
            index_parent_2 = max(index_1, index_2)

            first_seg_parent_1 = parent_1[:index_parent_1]

            mid_seg_parent_1 = parent_1[index_parent_1:index_parent_2 + 1]

            last_seg_parent_1 = parent_1[index_parent_2 + 1:]

            first_seg_parent_2 = parent_2[:index_parent_1]

            mid_seg_parent_2 = parent_2[index_parent_1:index_parent_2 + 1]

            last_seg_parent_2 = parent_2[index_parent_2 + 1:]

            # Creating childrens

            child_1 = np.concatenate((first_seg_parent_1, mid_seg_parent_2,
                                      last_seg_parent_1))

            child_2 = np.concatenate((first_seg_parent_2, mid_seg_parent_1,
                                      last_seg_parent_2))

        return child_1, child_2

    def mutation(self, child_1, child_2, prob_mutation=0.25):
        mutated_child_1 = np.empty((0, len(child_1)))

        t = 0
        for i in child_1:
            rand_num_to_mutate_or_not = np.random.rand()

            if rand_num_to_mutate_or_not < prob_mutation:
                if child_1[t] == 0:
                    mutated_child_1 = np.append(mutated_child_1, 1)
                else:
                    mutated_child_1 = np.append(mutated_child_1, 0)

                t += 1
            else:
                mutated_child_1 = np.append(mutated_child_1, child_1[t])

                t += 1

        mutated_child_2 = np.empty((0, len(child_2)))

        t = 0
        for i in child_2:
            rand_num_to_mutate_or_not = np.random.rand()

            if rand_num_to_mutate_or_not < prob_mutation:
                if child_2[t] == 0:
                    mutated_child_2 = np.append(mutated_child_2, 1)
                else:
                    mutated_child_2 = np.append(mutated_child_2, 0)

                t += 1
            else:
                mutated_child_2 = np.append(mutated_child_2, child_2[t])

                t += 1

        return mutated_child_1, mutated_child_2

    def start_optimization(self):
        x_y_string = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                               0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

        pool_of_solutions = np.empty((0, len(x_y_string)))
        best_of_a_generation = np.empty((0, len(x_y_string) + 1))
        best_values_of_generation = []

        for i in range(self.population):
            rd.shuffle(x_y_string)
            pool_of_solutions = np.vstack((pool_of_solutions, x_y_string))

        gen = 1

        start_time = time.time()  # start time (timing purposes)

        for i in range(self.generations):
            new_population = np.empty((0, len(x_y_string)))

            new_population_with_obj_val = np.empty((0, len(x_y_string) + 1))

            sorted_best = np.empty((0, len(x_y_string) + 1))

            print("------->Generation: #", gen)

            family = 1
            for j in range(int(self.population / 2)):
                print("--->Family: #", family)

                parent_1, parent_2 = self.find_parents_ts(pool_of_solutions, self.x, self.y)

                child_1, child_2 = self.crossover(parent_1, parent_2, self.prob_crsvr)

                mchild_1, mchild_2 = self.mutation(child_1, child_2, self.prob_mutation)

                # Fitness
                obj_val_mutated_child_1 = self.objective_value(self.x, self.y, mchild_1)[1]
                obj_val_mutated_child_2 = self.objective_value(self.x, self.y, mchild_2)[1]

                mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1, mchild_1))
                mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2, mchild_2))

                new_population = np.vstack((new_population, mchild_1, mchild_2))

                new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                         mutant_1_with_obj_val,
                                                         mutant_2_with_obj_val))
                family += 1

            pool_of_solutions = new_population
            sorted_best = np.array(sorted(new_population_with_obj_val,
                                          key=lambda x: x[0]))

            best_of_a_generation = np.vstack((best_of_a_generation, sorted_best[0]))
            print("Best of Generation #{}: {}".format(gen, sorted_best[0][0]))
            best_values_of_generation.append(sorted_best[0][0])

            gen += 1

        end_time = time.time()  # end time (timing purposes)

        sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                                 key=lambda x: x[0]))

        sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                                      key=lambda x: x[0]))

        sorted_last_population[:, 0] = 1 - (sorted_last_population[:, 0])
        sorted_best_of_a_generation[:, 0] = (sorted_best_of_a_generation[:, 0])

        best_string_convergence = sorted_last_population[0]

        best_string_overall = sorted_best_of_a_generation[0]

        print()
        print()
        print("------------------------------")
        print()
        print("Execution Time in Seconds:", end_time - start_time)  # exec. time

        final_solution_overall = self.objective_value(self.x, self.y, best_string_overall[1:])

        # the "svm_hp_opt.objective_value" function returns 3 things -->
        # [0] is the x value
        # [1] is the y value
        # [2] is the obj val for the chromosome (avg. error)
        print("Decoded C (Best):", round(final_solution_overall[0], 5))  # real value of x
        print("Accuracy (Best): ", round((final_solution_overall[1]), 5))  # obj val of final chromosome
        print()
        print("------------------------------")

        print("Values if not used the GA:")
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)

        model = svm.SVC(kernel="rbf", C=final_solution_overall[0])
        model.fit(x_train, np.ravel(y_train))

        filename = 'psych/trained_model_psych_ai.sav'
        pickle.dump(model, open(filename, 'wb'))


        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.20)

        model = svm.SVC(kernel="rbf")
        model.fit(x_train, np.ravel(y_train))

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy: ", accuracy)

train_df = pd.read_csv('first_csvs/psych.csv')

db_connection_str = 'mysql+pymysql://{}@{}/medintegral'.format(gv.USERNAME,gv.SERVER)
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM view_psych_ai_reviewed', con=db_connection)

if df.__len__() > 0:
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

    test_df = pd.concat([train_df, df], ignore_index=True, sort=False)

    objpo = Psych_AI(train_df,generations=1000)







