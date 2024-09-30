import pandas as pd
import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter
from numpy import array, float32, linspace
import seaborn as sns
from scipy.stats import norm, boxcox_normmax, boxcox, shapiro
from scipy.special import inv_boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.stats import shapiro 
from scipy. stats import lognorm
excel_data = pd.read_excel('dataset.xlsx', sheet_name='Worksheet')
df = pd.DataFrame({"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1]})
"""result = seasonal_decompose(df['цена нефти'], model='multiplicative', period=7)

plt.plot(df['цена нефти'], label='Original Time Series', color='blue')
data_without_seasonal = df['цена нефти'] / result.seasonal
plt.title('Air Passengers Time Series with and without Seasonal Component')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.legend()
plt.plot(data_without_seasonal, label='Original Data without Seasonal Component', color='green')
plt.show()"""
print(df['цена нефти'])
lmax_pearsonr_X, lmax_mle_X = boxcox_normmax(df['цена нефти'], method='all')    
print(f"lmax_pearsonr_X = {lmax_pearsonr_X}\t lmax_mle_X = {lmax_mle_X}\n")
X_boxcox = boxcox(df['цена нефти'], lmbda=(lmax_mle_X))
print(f"X_boxcox = {X_boxcox}\n")
Y_inverse_boxcox = inv_boxcox(X_boxcox, lmax_mle_X)
print(f"X_inverse_boxcox = {Y_inverse_boxcox}\n")
print(shapiro(df['цена нефти']))

sns.kdeplot(df["цена нефти"])
plt.ylabel("Частота")
plt.title("Частотное распределение")
plt.show()
"""sns.lineplot(
    x="года",
    y="цена нефти",
    data=df, legend=True)
b =[]
for i in range(0, 1044):
    if i%104==0: b.append(i)
plt.xticks(b)
plt.show()"""