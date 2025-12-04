import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from limpiar_datos import limpiar_datos

df = pd.read_csv('dataset_combinado.csv')
limpiar_datos(df)
df_filtrado = df.copy()
df_filtrado['TASA_IRREGULARIDAD'] = df_filtrado['RRAA_IRREGULAR'] / (df_filtrado['RRAA_TOTAL'] + 1e-10)
df_filtrado['TIPO_MIGRACION'] = (df_filtrado['TASA_IRREGULARIDAD'] > 0.5).astype(int)  # 1=IRREGULAR, 0=REGULAR

feature_cols = ["SEXO", "EDAD_NUMERICA", "PAIS_CODIGO", "AÑO", "CODREGEO", "CENSO AJUSTADO", "INFLACION", "CRECIMIENTO_PIB", "DESEMPLEO"]
label_col = "TIPO_MIGRACION"          # o la variable que quieras predecir

labels = df_filtrado.pop(label_col)
features = df_filtrado[feature_cols]

model = Sequential([
    Dense(32, activation='relu', input_shape=(len(feature_cols),)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(features, labels, epochs=30)

#https://www.tensorflow.org/tutorials/load_data/pandas_dataframe?hl=en
#https://www.kaggle.com/code/manuelalb/ejemplo-de-pandas-red-neuronal