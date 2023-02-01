import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.neighbors import  KNeighborsClassifier
from sklearn import  metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict



df = pd.read_csv('glass.csv')

sns.countplot(x = 'Type', data=df)
plt.show()

print(df.head(n=20))
print(df.describe())
print(df.isnull().sum())

x = df.drop(['Type'], axis=1)

# # Матриця кореляції  

# plt.figure(figsize=(20, 10))
# ax = sns.heatmap(df.corr(), annot = True ,cmap='viridis')
# plt.show()

# Попарне порівняння

# p_plot = sns.pairplot(df, hue='RI')
# plt.show()

# #########################################################

scaler = StandardScaler()

scaler.fit(df.drop('Type', axis = 1))

scaled = scaler.transform(df.drop('Type', axis = 1))

df_scaled =  pd.DataFrame(scaled, columns = df.columns[:-1])

X = df_scaled
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=202)

# #########################################################

# knn = KNeighborsClassifier(n_neighbors= 3, metric='minkowski')

knn = KNeighborsClassifier(n_neighbors= 3, metric='cosine')

# knn = KNeighborsClassifier(n_neighbors= 3, metric='euclidian')

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print(classification_report(y_test,prediction))

print(metrics.accuracy_score(y_test, prediction))

# #########################################################

CLF = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(200), activation='relu', solver="adam").fit(X_train, y_train)
print(CLF.score(X_test, y_test))

