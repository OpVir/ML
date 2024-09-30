from scipy.stats import  shapiro
import pandas as pd
from numpy import array

#Получение данных из файла
excel_data = pd.read_excel('dataset.xlsx', sheet_name='Worksheet')
df = pd.DataFrame({"цена нефти":array(excel_data['Цена барреля ₽'])[::-1], "года":array(excel_data['Дата'])[::-1]})

#использование функции оценки критерием Шапиро-Уилка
print(shapiro(df['цена нефти']))
#Вывод: ShapiroResult(statistic=0.9426260428485587, pvalue=1.129785803072025e-19)