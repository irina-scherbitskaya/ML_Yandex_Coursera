'''
Загрузите выборку Wine по адресу 
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 

Извлеките из данных признаки и классы. 
Класс записан в первом столбце (три варианта), признаки —
в столбцах со второго по последний.
Более подробно о сути признаков можно прочитать по адресу
https://archive.ics.uci.edu/ml/datasets/Wine

Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). 
Создайте генератор разбиений, который перемешивает выборку перед формированием 
блоков (shuffle=True).

Для воспроизводимости результата, создавайте генератор KFold с фиксированным 
параметром random_state=42. В качестве меры качества используйте долю верных 
ответов (accuracy).

Найдите точность классификации на кросс-валидации для метода k ближайших
соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При 
каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)?
Данные результаты и будут ответами на вопросы 1 и 2.

Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
Снова найдите оптимальное k на кросс-валидации.

Какое значение k получилось оптимальным после приведения признаков к одному масштабу? 
Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?
'''
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing


def test(kf, x, y):
    arr = []
    for i in range(1, 51):
        kn = KNeighborsClassifier(i)
        arr.append([cross_val_score(kn, x, y, cv=kf, scoring='accuracy').mean(), i])
    print(sorted(arr, key=lambda x: -x[0])[0])


data = pd.read_csv('/Users/irinascherbitskaya/Downloads/wine.data', header=None)
y = data[0]
x = data.loc[:, 1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test(kf, x, y)
x = preprocessing.scale(x)
test(kf, x, y)