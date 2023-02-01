import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import xgboost as xg

# from pandas.plotting import scatter_matrix


vine = pd.read_csv('./winequality-red.csv')

# Читаємо основні відомості про датасет
print(vine.head())
vine.info()
print(vine.describe())

# Графік, який відсоток інформації відсутній, для кожної з колонок. 
vine.isnull().mean().plot.bar(figsize=(12,6))
plt.ylabel('Percentage of missing values') 
plt.xlabel('Features') 
plt.title('Missing Data in Percentages')
plt.show()

# Шукаємо нульові значення і заповнюємо середнім
vn_null_columns = vine.isnull().sum()
for (index, value) in vn_null_columns.iteritems():
    if value != 0:
        vine[index] = vine[index].fillna(vine[index].mean())

# Перевіряємо, чи зповнились пусті значення в датасеті
vine.info()

# Перевіряємо чи не сильний у нас відхил між середнім кожної колонки і її медіаною, якщо відхил суттєвий,
# то бажано почистити естримальні значення в датасеті
for column in vine.columns:
    print("Column name: {0}. Diff: {1}".format(column, abs(vine[column].mean() - vine[column].median()))) 


# Хітмап кореляції змінних
plt.figure(figsize=(vine.shape[1],vine.shape[1]))
matrix = np.triu(vine.corr())
sns.heatmap(vine.corr(), annot=True, linewidth=.8, mask=matrix, cmap="rocket")
plt.show()

# Скейлювання даних (підгін під тренування)
scaler = MinMaxScaler()
# Підготовка вхідних-вихідних даних
X = vine.drop('quality', axis=1)
X = scaler.fit_transform(X.values)
Y = vine[['quality']]
Y = scaler.fit_transform(Y.values)

# Перевірка коректності
print("X shape", X.shape)
print("Y shape:", Y.shape)

# Спліт на тренувальну і тестову дату
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
random_state=100)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
print("X_test shape:", X_train.shape)
print("Y_test shape:", Y_train.shape)

# Тренування
lr = LinearRegression()
lr.fit(X_train, Y_train)
YPr_train = lr.predict(X_train)
YPr = lr.predict(X_test)
print("Train score:", lr.score(X_train, Y_train))
print("Test score:", lr.score(X_test, Y_test))

# Коефіцієнти
print(lr.coef_) 
print(lr.intercept_) 

# Вираховування похибки
MSE1 = np.square(np.subtract(Y_test,YPr)).mean() 
print("RMSE1 : % f" %(MSE1))

print("Train score:", lr.score(X_train, Y_train))
print("Test score:", lr.score(X_test, Y_test))


X, y = vine.iloc[:, :-1], vine.iloc[:, -1]
  
# Спліт
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 111)
  
# Використання XGBR
xgbr = xg.XGBRegressor(objective ='reg:squarederror',n_estimators = 10, seed = 111)
  
# Натягування на модель
xgbr.fit(train_X, train_y)
  
# Передбачення
pred = xgbr.predict(test_X)


  
# Вираховування похибки 
MSE2 = np.square(np.subtract(test_y,pred)).mean()
print("RMSE2: % f" %(MSE2))

print("Train score:", xgbr.score(train_X, train_y))
print("Test score:", xgbr.score(test_X, test_y))