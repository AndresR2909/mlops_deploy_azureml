# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import mlflow
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

def init():
    global model_classification

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder
    # Please provide your model's folder name if there's one

    model_classification = extract_registered_model()

def extract_registered_model():

    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    print("path del modelo clasificación: ", model_path)
    model = mlflow.xgboost.load_model(model_path)

    print("Se ha cargado el modelo de clasificación")

    return model

class DataCleaning:
    #Inicializamos cargando la data
    def __init__(self, data_path = None, data_set = None ) -> None:
        if data_set is None:
            self.data_path = data_path
            self.dataframe = pd.read_csv(data_path, delimiter=";")
        else: 
            self.dataframe = data_set

    #Creamos un método para manejar los valores que estén nulos
    def handle_missing_values(self):
        self.dataframe = self.dataframe.dropna() #Método para eliminar las filas con valores faltantes de (ojo!! pueden interpolarse o rellenarse)

    #Creamos un método para manejar variables que no son numéricas
    def handle_categorical_features(self):
        df_copy = self.dataframe.copy()

        for col in df_copy.columns:
            if df_copy[col].dtype == object:
                le = LabelEncoder()

                df_copy[col] = le.fit_transform(df_copy[col])

        self.dataframe =  df_copy
    
    #Creamos un método para manejar la clase objetivo de manera desbalanceada (si lo está)
    def handle_imbalanced_data(self, objetive):
        pass

    #Creamos un método para escalar las caracteristicas que queramos
    def scalate_features(self, feature):
        pass

    #Creamos un método para identificar y eliminar caracteristicas que sean irrelevantes
    def remove_irrelevant_features(self):
        pass

    #Creamos un método para obtener X y y test, para evaluación de performance
    def performance_split(self, target_feature):
        self.X_test_performance = self.dataframe.drop(columns=[target_feature])
        self.y_test_performance = self.dataframe[target_feature]
    
    #Creamos un método para realizar el split de la data
    def split_data(self, target_feature, size):
        X = self.dataframe.drop(columns=[target_feature])
        y = self.dataframe[target_feature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=size, random_state=42)

def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    print("Iniciando la ejecución de clasificación")
    print("modelo de clasificación", model_classification)
    
    result_list = []

    for file_path in mini_batch:
        df = pd.read_csv(file_path, delimiter=";")

        # Realizar las predicciones
        data_clean = DataCleaning(data_set = df)
        data_clean.handle_missing_values()
        data_clean.handle_categorical_features()

        column_find = "Diabetes"
        if column_find in df.columns:
            # Respuesta de evaluación de performance
            data_clean.performance_split("Diabetes")
            pred = model_classification.predict(data_clean.X_test_performance)
            accuracy = accuracy_score(data_clean.y_test_performance, pred)
            recall = recall_score(data_clean.y_test_performance, pred)
            f1 = f1_score(data_clean.y_test_performance, pred)

            result_list.append(json.dumps({'accuracy': accuracy,
                            'recall': recall,
                            'f1':f1}))
        else:
            # Respuesta de predicciones
            pred = model_classification.predict(data_clean.dataframe)
            pred_converted = np.where(pred == 1, 'diabetes', 'no diabetes')
            # Retornar las predicciones en formato JSON
            result_list.append({'predictions': pred_converted.tolist()})

        

    return result_list