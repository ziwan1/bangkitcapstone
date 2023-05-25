import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

dataset_dir = "./Dataset/"
files_list = []

for path in os.listdir(dataset_dir):
  if os.path.isfile(os.path.join(dataset_dir, path)):
    files_list.append(path)

print(files_list)

for datasets in files_list:
  df = pd.read_csv(f'./Dataset/{datasets}')
  df.index = pd.to_datetime(df['time'], format = '%Y-%m-%dT%H:%M') # Ubah ke format datetime standar

  main_cols = ['temperature_2m (°C)', 'relativehumidity_2m (%)',
              'apparent_temperature (°C)','precipitation (mm)', 
              'rain (mm)','cloudcover (%)','shortwave_radiation (W/m²)',
              'direct_radiation (W/m²)', 'diffuse_radiation (W/m²)',
              'direct_normal_irradiance (W/m²)']

  tz = pytz.timezone('Asia/Singapore')
  for times in df['time']:
    times = datetime.fromtimestamp(times, tz).isoformat()

  temp = df['temperature_2m (°C)']
  temp_df = pd.DataFrame({'Temperature':temp})
  temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)

  day = 60*60*24
  year = 365.2425*day

  temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2*np.pi / day))
  temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2* np.pi / day))
  temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2*np.pi / year))
  temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2*np.pi / year))

  temp_df = temp_df.drop('Seconds', axis=1)

  p_temp_df = pd.concat([df['relativehumidity_2m (%)'], df['apparent_temperature (°C)'], 
                        df['precipitation (mm)'], df['rain (mm)'], 
                        df['cloudcover (%)'], df['shortwave_radiation (W/m²)'],
                        df['direct_radiation (W/m²)'], df['diffuse_radiation (W/m²)'],
                        df['direct_normal_irradiance (W/m²)'], temp_df], axis = 1)

  def df_to_X_y(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
      row = [var for var in df_as_np[i:i+window_size]]
      X.append(row)
      label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1],
              df_as_np[i+window_size][2], df_as_np[i+window_size][3],
              df_as_np[i+window_size][4], df_as_np[i+window_size][5],
              df_as_np[i+window_size][6], df_as_np[i+window_size][7],
              df_as_np[i+window_size][8], df_as_np[i+window_size][9]]
      y.append(label)
    return np.array(X), np.array(y)

  X, y = df_to_X_y(p_temp_df)

  X_train, y_train = X[:70000], y[:70000]
  X_val, y_val = X[70000:80000], y[70000:80000]
  X_test, y_test = X[80000:], y[80000:]

  means_list = []
  stds_list = []

  for i in range(10):
    means_list.append(np.mean(X_train[:, :, i]))
    stds_list.append(np.std(X_train[:, :, i]))

  def preprocess(X):
    for i in range(10):
      X[:, :, i] = (X[:, :, i] - means_list[i]) / stds_list[i]

  def preprocess_output(y):
    for i in range(10):
      y[:, i] = (y[:, i] - means_list[i]) / stds_list[i]

  preprocess(X_train)
  preprocess(X_val)
  preprocess(X_test)
  preprocess_output(y_train)
  preprocess_output(y_val)
  preprocess_output(y_test)

  model2 = Sequential()
  model2.add(InputLayer((7,14)))
  model2.add(LSTM(128))
  model2.add(Dense(16, 'relu'))
  model2.add(Dense(10, 'linear'))

  checkpoint = ModelCheckpoint(f'model_{datasets}/', save_best_only=True)
  model2.compile(
      loss=MeanSquaredError(), 
      optimizer=Adam(learning_rate=0.0001), 
      metrics=[RootMeanSquaredError()])

  model2.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=30, callbacks=[checkpoint])

  def plot_predictions2(model, X, y, start=0, end=300):
    predictions = model.predict(X)
    y1_preds, y2_preds, y3_preds, y4_preds, y5_preds, y6_preds, y7_preds, y8_preds, y9_preds, y10_preds = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4], predictions[:, 5], predictions[:, 6], predictions[:, 7], predictions[:, 8], predictions[:, 9]
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7], y[:, 8], y[:, 9]
    df = pd.DataFrame(data = {'R.Humidity Predictions': y1_preds,
                              'R.Humidity Actuals': y1,
                              'Ap.Temp Predictions': y2_preds,
                              'Ap.Temp Actuals': y2,
                              'Precipitation Predictions': y3_preds,
                              'Precipitation Actuals': y3,
                              'Rain Predictions': y4_preds,
                              'Rain Actuals': y4,
                              'Cloud Cover Predictions': y5_preds,
                              'Cloud Cover Actuals': y5,
                              'Shortwave Rad. Predictions': y6_preds,
                              'Shortwave Rad. Actuals': y6,
                              'Direct Rad. Predictions': y7_preds,
                              'Direct Rad. Actuals': y7,
                              'Diffuse Rad. Predictions': y8_preds,
                              'Diffuse Rad. Actuals': y8,
                              'Direct Norm. Irradiance Pred': y9_preds,
                              'Direct Norm. Irradiance Actuals': y9,
                              'Temperature Predictions': y10_preds,
                              'Temperature Actuals': y10})
    
    for i in df.columns:
      plt.plot(df[i][start:end])

    df.to_csv(f'predictions_{datasets}.csv', index=False)
    return df[start:end]

  plt.figure(figsize=(50,20))
  plot_predictions2(model2, X_test, y_test)

  preds = pd.read_csv(f'predictions_{datasets}.csv')
  list_cols = list(preds.columns)
  for i in range(len(list_cols) // 2):
      print(f"R2-SCORE VARIABEL {list_cols[2*i]} dan {list_cols[2*i+1]}: {r2_score(preds[list_cols[2*i + 1]], preds[list_cols[2*i]])}")
      plt.figure(figsize=(40,20))
      plt.title(f"{list_cols[2*i]} vs {list_cols[2*i + 1]}", fontsize=40)
      plt.plot(preds[list_cols[2*i][0:100]])
      plt.plot(preds[list_cols[2*i+1][0:100]])
      plt.xlabel("Time", fontsize=25)
      plt.ylabel("Predicted Value", fontsize=25)
      plt.grid()
      plt.savefig(f"Dataset {datasets} - {list_cols[2*i]} vs {list_cols[2*i + 1]}.png")
