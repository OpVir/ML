from scipy.stats import  boxcox_normmax, boxcox
from scipy.special import inv_boxcox
from numpy import array
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

#Получение данных из файла
excel_data = pd.read_excel('dataset.xlsx', sheet_name='Worksheet')
df = pd.DataFrame({"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1]})

#Вычесление оптемально параметра лямбды
print(df['цена нефти'])
lmax_pearsonr_X, lmax_mle_X = boxcox_normmax(df['цена нефти'], method='all')    

#Прямое преобразование
X_boxcox = boxcox(df['цена нефти'], lmbda=(lmax_mle_X))

#Обратное преобразование
Y_inverse_boxcox = inv_boxcox(X_boxcox, lmax_mle_X)
sns.kdeplot(X_boxcox)
plt.ylabel("Частота")
plt.title("Частотное распределение")
plt.show()