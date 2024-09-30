import pandas as pd
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout, InputLayer, BatchNormalization, Flatten, Input
from keras.activations import tanh, relu
from numpy import array, reshape, float32
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import  boxcox_normmax, boxcox
import random
import keras_tuner
import numpy
# Загрузка данных из файла Excel
counter = 0

def get_dataset(train_size, xlsx, worksheet):
    excel_data = pd.read_excel(f"{xlsx}.xlsx", sheet_name=worksheet)
    df = {"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1], "цена доллара":array(excel_data['Цена доллара к рублю'])[::-1]}
    df = [df['цена доллара'][0:train_size:],df['цена доллара'][train_size:-4:],df['цена нефти'][0:train_size:],df['цена нефти'][train_size:-4:]]
    return df

def formater_for_rub(x, pos):
    return '{:f} rub'.format(x)

def boxcox_data_formater(df):
    lmax_pearsonr, lmax_mle = boxcox_normmax(df, method='all')    
    result = boxcox(df, lmbda=(lmax_mle))
    return result

def Normilize_with_MinMax(data):
    data = data.reshape(-1, 1)
    minmaxScaler = MinMaxScaler(feature_range=(0, 1))
    fit = MinMaxScaler.fit(minmaxScaler, X=data)
    res = MinMaxScaler.fit_transform(minmaxScaler, X=data)
    return res

def croptdata_formater(data, size):
    result = []
    if (len(data)%size==0):
        p = len(data)//size
    else: p = len(data)//size+1
    for i in range(0, p):
        if i+size<len(data):
            result.append(data[size*i:size*i+size:])
        else:
            result.append([data[i::]])
    return result
#TODO
def print_plt(metric, predicate_data, real_data):
    figure, ax = plt.subplots(2, 2) 
    global counter
    ax[0][0].plot(real_data, "black")
    ax[0][0].plot(predicate_data[0].reshape(-1, 1), "blue")
    ax[0][0].set_title("Нет/Нет")
    ax[0][0].legend(["Реальные данные","Предсказанные данные"]) 
    
    ax[0][1].plot(real_data, "black")
    ax[0][1].plot(predicate_data[1].reshape(-1, 1), "blue")
    ax[0][1].set_title("Нет/BoxCox")
    ax[0][1].legend(["Реальные данные","Предсказанные данные"]) 
    
    ax[1][0].plot(real_data, "black")
    ax[1][0].plot(predicate_data[2].reshape(-1, 1), "blue")
    ax[1][0].set_title("Нормализация/Нет")
    ax[1][0].legend(["Реальные данные","Предсказанные данные"]) 
    
    ax[1][1].plot(real_data, "black")
    ax[1][1].plot(predicate_data[3].reshape(-1, 1), "blue")
    ax[1][1].set_title("Нормализация/BoxCox")
    ax[1][1].legend(["Реальные данные","Предсказанные данные"]) 
    
    plt.title(counter)
    plt.show()
    
    counter += 1

def model_predicate(model, data):
    metric = model.evaluate(data[0], data[1], batch_size=20)
    predicate = model.predict(data[0])
    return (metric, predicate)

def build(hp):
        model = Sequential([Input(shape=(1, ))])
        for i in range(hp.Int(name="num_layer", min_value=1, max_value=3)):
            model.add(Dense(
                units=hp.Int(f"unit_{i}", min_value=10, max_value=64, step=2),
                activation=hp.Choice("activation", ["relu", "tanh", "leaky_relu"])
            ))
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.25))
        model.add(Dense(1, "linear"))
        

        model.compile(
            optimizer=Adam(),
            loss="mse",
            metrics=["mae"],
        )
        return model

        

tuner = keras_tuner.RandomSearch(
    hypermodel=build,
    objective="val_loss",
    max_trials=20,
    executions_per_trial=10,
    overwrite=False,
    directory="my_dir",
    project_name="helloworld",
)



data = get_dataset(900, "dataset", "Worksheet")
#разберись с кроссвалидацией
#тут убери решапе, пусть в будет в fit
figure, ax = plt.subplots(2, 2) 
result = []
for i in 0,1:
    for j in 0,1:
        x= data[0]
        y = data[2]
        x_val = data[1]
        y_val= data[3]
        a="Нет"
        b="Нет"
        if (j==1):
            b="BoxCox"
            x = boxcox_data_formater(x)
            y = boxcox_data_formater(y)
            x_val = boxcox_data_formater(x_val)
            y_val = boxcox_data_formater(y_val)
        if (i==1):
            a="Нормализация"
            x = Normilize_with_MinMax(x).reshape(-1, 1)
            y = Normilize_with_MinMax(y).reshape(-1, 1)
            x_val = Normilize_with_MinMax(x_val).reshape(-1, 1)
            y_val = Normilize_with_MinMax(y_val).reshape(-1, 1)
        tuner.search(x = x, y = y, validation_data=(x_val, y_val))
        models = tuner.get_best_models(num_models=1)
        best_model = models[0]
        result.append( models[0])
        best_model.summary()
        metric, predicate_data= model_predicate(best_model,  (x_val, y_val)) #сделай эту херь
        

        ax[i][j].plot(y_val, "black")
        ax[i][j].plot(predicate_data.reshape(-1, 1), "blue")
        ax[i][j].set_title(f"{a}/{b}")
        ax[i][j].legend(["Реальные данные","Предсказанные данные"]) 
plt.show()
print("\n\n---------------------------------------\n\n")







 

