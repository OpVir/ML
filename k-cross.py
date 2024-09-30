import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pylab as plt
from keras.utils import timeseries_dataset_from_array


def get_dataset( xlsx, worksheet):
    excel_data = pd.read_excel(f"{xlsx}.xlsx", sheet_name=worksheet)
    df = {"цена нефти":np.array(excel_data['Цена барреля ₽'])[::-1], "года":np.array(excel_data['Дата'])[::-1], "цена доллара":np.array(excel_data['Цена доллара к рублю'])[::-1]}
    df = [df['цена доллара'],df['цена нефти']]
    return df

data = get_dataset("dataset","Worksheet")
X, y = data[0], data[1]
kf = KFold(n_splits=10)
print(kf.split(X))
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    data_train = timeseries_dataset_from_array(data=X_train, targets=y_train, sequence_length=10)
    data_test = timeseries_dataset_from_array(data=X_train, targets=y_train, sequence_length=10)
    for batch in data_train:
        print(batch)