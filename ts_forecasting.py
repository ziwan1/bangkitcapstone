import tensorflow as tf
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

parent_dir = 'Dataset'
list_dir_arr =  []
list_filename_arr =  []

# num_lines_to_delete = 3
# def delete_lines(filename, num_lines):
#     with open(filename, 'r') as file:
#         lines = file.readlines()

#     with open(filename, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(csv.reader(lines[num_lines:]))

#     print(f"The first {num_lines} line(s) have been deleted from {filename}.")

for i in os.listdir(parent_dir):
    dir_name = os.path.join(parent_dir, i)
    for csv_file_in_this_dir in os.listdir(dir_name):
        # print(csv_file_in_this_dir)
        # print(os.path.join(parent_dir, i, csv_file_in_this_dir))
        # print(os.path.join(parent_dir, i, csv_file_in_this_dir))
        list_dir_arr.append(os.path.join(parent_dir, i, csv_file_in_this_dir))
        csv_file_in_this_dir = csv_file_in_this_dir.replace(" ","_")
        xi = i.replace(" ","_")
        list_filename_arr.append(f"{xi}_{csv_file_in_this_dir[:-4]}")
        # print(f"file {csv_file_in_this_dir[:-4]}.csv on {dir_name} is appended")
        # delete_lines(os.path.join(parent_dir, i, csv_file_in_this_dir), num_lines_to_delete)

# for z in list_dir_arr:
#     print(z)
for dir_name in list_filename_arr:
    os.makedirs(f"model_result/{dir_name}", exist_ok=True)

for iii, data_dir in enumerate(list_dir_arr):
    try:
        start_time = time.perf_counter()
    except:
        print("error")
    print(f"\nCurrent file: {list_filename_arr[iii]}\n")
    df = pd.read_csv(data_dir)

    tz = pytz.timezone('Asia/Singapore')
    time_iso = []
    for times in df['time']:
        time_iso.append(datetime.fromtimestamp(times, tz).isoformat())

    df['time'] = time_iso
    df.index = pd.to_datetime(df['time'], format = '%Y-%m-%dT%H:%M:00+08:00')

    main_cols = ['temperature_2m', 'relativehumidity_2m',
             'apparent_temperature','precipitation', 
             'rain','cloudcover','shortwave_radiation',
             'direct_radiation', 'diffuse_radiation',
             'direct_normal_irradiance']


    main_df = df[main_cols]

    main_df['seconds'] = main_df.index.map(pd.Timestamp.timestamp)

    day =  60*60*24
    year = 365.2425*day

    main_df['sin(day)'] = np.sin(main_df['seconds'] * (2*np.pi/day))
    main_df['cos(day)'] = np.cos(main_df['seconds'] * (2*np.pi/day))
    main_df['sin(year)'] = np.sin(main_df['seconds'] * (2*np.pi/year))
    main_df['cos(year)'] = np.cos(main_df['seconds'] * (2*np.pi/year))

    main_df.drop(columns='seconds', inplace=True)

    def df_to_X_y(df, win_size = 7):
        df_as_np = df.to_numpy()
        X, y = [], []

        for i in range(len(df_as_np) - win_size):
            row = [x for x in df_as_np[i:i+win_size]]
            X.append(row)
            label = [df_as_np[i+win_size][0], df_as_np[i+win_size][1],
                    df_as_np[i+win_size][2], df_as_np[i+win_size][3],
                    df_as_np[i+win_size][4], df_as_np[i+win_size][5],
                    df_as_np[i+win_size][6], df_as_np[i+win_size][7],
                    df_as_np[i+win_size][8], df_as_np[i+win_size][9]]
            y.append(label)
        
        return np.array(X), np.array(y)
    
    X, y = df_to_X_y(main_df)

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
    
    def re_preprocess_y(y, index):
        return (y * stds_list[index]) + means_list[index]
    
    preprocess(X_train)
    preprocess(X_val)
    preprocess(X_test)
    preprocess_output(y_train)
    preprocess_output(y_val)
    preprocess_output(y_test)

    model = Sequential()
    model.add(InputLayer((7,14)))
    model.add(LSTM(128)) #64
    model.add(Dense(16, 'relu'))
    model.add(Dense(10, 'linear'))

    model.summary()

    checkpoint = ModelCheckpoint(f'model_result/{list_filename_arr[iii]}/', save_best_only=True)
    model.compile(
        loss=MeanSquaredError(), 
        optimizer=Adam(learning_rate=0.0001), 
        metrics=[RootMeanSquaredError()])
    
    model.fit(X_train, y_train, 
          validation_data = (X_val, y_val), 
          epochs=30, 
          callbacks=[checkpoint])

    start_date = "2023-04-11 00:00:00"
    end_date = "2025-04-10 23:00:00"
    time = pd.date_range(start_date, end_date, freq="H")
    palette = pd.DataFrame(time, columns=['time'])

    palette.index = pd.to_datetime(palette['time'], format='%Y-%m-%d %H:%M:%S')
    palette['seconds'] = palette.index.map(pd.Timestamp.timestamp)

    palette.drop(columns='time', inplace=True)

    palette['sin(day)'] = np.sin(palette['seconds'] * (2*np.pi/day))
    palette['cos(day)'] = np.cos(palette['seconds'] * (2*np.pi/day))
    palette['sin(year)'] = np.sin(palette['seconds'] * (2*np.pi/year))
    palette['cos(year)'] = np.cos(palette['seconds'] * (2*np.pi/year))
    palette.drop(columns='seconds', inplace=True)

    def construct_lt_prediction(X, model, palette):
        preds = []
        for i in range(10000):
            latest_pred = model.predict(X[-1:], verbose=0)
            last_data = X[-1].tolist()
            new_palette = [last_data[1:] + [[latest_pred[0][0], latest_pred[0][1], latest_pred[0][2], latest_pred[0][3], latest_pred[0][4], latest_pred[0][5], latest_pred[0][6], latest_pred[0][7], latest_pred[0][8], latest_pred[0][9], palette[palette.columns[0]][i], palette[palette.columns[1]][i], palette[palette.columns[2]][i], palette[palette.columns[3]][i]]]]
            X = np.array(X.tolist() + new_palette)
            print(f"Proses {i}/10000 berjalan...")
            preds.append(latest_pred)
        print("Proses selesai")
        return X, preds
    
    X_pred_new, predictions = construct_lt_prediction(X_test, model, palette)

    prediction_arr = []

    for i in range(len(predictions)):
        prediction_arr.append(predictions[i][0].tolist())
    
    forecast_df = pd.DataFrame(prediction_arr, columns=main_df.columns[:-4])

    forecast_df['time'] = palette.reset_index()['time'][0:10000]

    for cols in range(len(forecast_df.columns[:-1])):
        index = 0
        plt.figure(figsize=(20, 5))
        val = forecast_df[forecast_df.columns[cols]].to_numpy()
        plt.plot(re_preprocess_y(val, index), color='green')
        plt.title(f"Forecast of {forecast_df.columns[cols]}", fontsize=20)
        plt.xlabel("Data", fontsize=15)
        plt.ylabel(f"{forecast_df.columns[cols]}", fontsize=15)
        plt.savefig(f"model_result/{list_filename_arr[iii]}/forecast_{cols}.png")
        index += 1

    # plt.savefig(f"model_result/{list_filename_arr[iii]}/{list_filename_arr[iii]}_forecast_plot.png")
    try:
        end_time = time.perf_counter()  # Record the ending time

        elapsed_time = end_time - start_time  # Calculate the elapsed time

        print(f"Elapsed time on this loop: {elapsed_time} seconds")
    except:
        print("error 2")
    # print("Elapsed time on this loop: {:.2f} seconds".format(elapsed_time))
    # break
