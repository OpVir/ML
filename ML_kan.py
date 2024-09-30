from kan import *
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LSTM, Dropout, InputLayer, BatchNormalization, Flatten, Input
from keras.activations import tanh, relu
from numpy import array, reshape, float32, shape
from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import  boxcox_normmax, boxcox
import random
import keras_tuner
import numpy
from torch import *

def get_dataset(train_size, xlsx, worksheet):
    excel_data = pd.read_excel(f"{xlsx}.xlsx", sheet_name=worksheet)
    df = {"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1], "цена доллара":array(excel_data['Цена доллара к рублю'])[::-1]}
    df = [df['цена доллара'][0:train_size:],df['цена доллара'][train_size:-4:]]
    return df


data = get_dataset(900, "dataset", "Worksheet")
dataset = dataset = create_dataset(data, n_var=2)
print(dataset.shape)
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)

model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);