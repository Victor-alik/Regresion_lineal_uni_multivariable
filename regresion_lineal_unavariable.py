#%%
#Regresion lineal
from sklearn.datasets import load_boston 
import pandas as pd 

 #cargamos los datos 
data=load_boston()
df=pd.DataFrame(data.data, columns=data.feature_names)
#%%
#Variables de respuesta
df['MEDV']=data.target[df.index]
df.head()
#%%
df.shape
#%%
descripcion=data.DESCR 
print(descripcion[148:1225])
#%%
df.corr()['MEDV'].sort_values()#para calcular la correolacion respecto de una variable en especifico (MEDV)
#%%
#Regrecion con una variable
X=df['LSTAT'].values.reshape(-1,1)#-1=numero de filas indefinido
y=df['MEDV'].values.reshape(-1,1)#1=numero de columnas=uno
#%%
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=100)
#%%
X_train[:5]
#%%
from sklearn.linear_model import LinearRegression 
#creamos el modelo y lo entrenmos
reg=LinearRegression().fit(X_train,y_train)
#Predicciones valores de entrenamiento
y_train_hat=reg.predict(X_train)
#Predicciones de valores de validacion
y_test_hat=reg.predict(X_test)
#%%
import matplotlib.pyplot as plt
import numpy as np 
#creamos scatter plot de los datos de entrenamiento
plt.scatter(X_train,y_train)
#Creamos scatter plot de los datos de validacion
plt.scatter(X_test,y_test)
# En X_plot guardamos valores de entre 0 y 40
X_plot=np.linspace(0,40).reshape(-1,1)
#Con el modelo predecimos X_plot
y_plot=reg.predict(X_plot)
#Graficamos el modelo
plt.plot(X_plot, y_plot, 'r--')
#%%
from sklearn.metrics import r2_score 
print('Entrenamiento', r2_score(y_train,y_train_hat))
print('Prueba',r2_score(y_test,y_test_hat))#Son valores muy bajos, no es suficiente
#0.54
#0.53
#%%
####Regrecion multiple.###
#obtenemos las varibales
X=df.drop('MEDV',axis=1)
y=df['MEDV'].values.reshape(-1,1)

#hacemo split del 33%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=100)

#Entrenamos al modelo
reg=LinearRegression().fit(X_train,y_train)

#realizamos las predicciones
y_train_hat=reg.predict(X_train)
y_test_hat=reg.predict(X_test)

#Calculamos el error
print('Entrenamieno',r2_score(y_train,y_train_hat))
print('Prueba',r2_score(y_test,y_test_hat))

#aumento el R2 de entrenamiento, pero el de validacion disminuyo
