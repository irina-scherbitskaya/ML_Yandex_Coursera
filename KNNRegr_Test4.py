'''
Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
а целевой вектор — в поле target.

Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.

Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).

Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр
добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.

В качестве метрики качества используйте среднеквадратичную ошибку
(параметр scoring='mean_squared_error' у cross_val_score; при использовании библиотеки
scikit-learn версии 0.18.1 и выше необходимо указывать scoring='neg_mean_squared_error').
  
Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам
с random_state = 42, не забудьте включить перемешивание выборки (shuffle=True).
Определите, при каком p качество на кросс-валидации оказалось оптимальным.
 Обратите внимание, что cross_val_score возвращает массив показателей качества
 по блокам; необходимо максимизировать среднее этих показателей. Это значение параметра и будет ответом на задачу.
 '''
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing

boston = load_boston()
x = boston.data
y = boston.target
x = preprocessing.scale(x)

arr = []
kf = KFold(n_splits=5, shuffle=True,random_state=45)

for i in range(1000, 10001, 46):
    kn = KNeighborsRegressor(n_neighbors=5, weights='distance', metric = 'minkowski', p = i/1000)
    arr.append([cross_val_score(kn, x, y, cv=kf,scoring='neg_mean_squared_error').mean(), i/1000])
print(sorted(arr, key = lambda x: -x[0])[0])