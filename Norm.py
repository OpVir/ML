from sklearn.preprocessing import MinMaxScaler
from scipy.stats import  boxcox_normmax, boxcox
from sklearn.model_selection import KFold
from scipy.special import inv_boxcox
from numpy import array, reshape
import pandas as pd

#Получение данных из файла
excel_data = pd.read_excel('dataset.xlsx', sheet_name='Worksheet')
df = {"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1]}

#Вычесление оптемально параметра лямбды
lmax_pearsonr_X, lmax_mle_X = boxcox_normmax(df['цена нефти'], method='all')    

#Прямое преобразование
X_boxcox = boxcox(df['цена нефти'], lmbda=(lmax_mle_X))

#Обратное преобразование
X_inverse_boxcox = inv_boxcox(X_boxcox, lmax_mle_X)

minmaxScaler = MinMaxScaler(feature_range=(0, 1))
result = MinMaxScaler.fit_transform(minmaxScaler, X=X_boxcox.reshape(-1, 1))
