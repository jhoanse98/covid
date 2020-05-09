# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:04:53 2020

@author: LENOVO
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url="https://api.covid19api.com/total/dayone/country/colombia"
response=requests.get(url)
json_response=response.json()
colombia=json_response



#analisis para el país de colombia

data= pd.DataFrame(colombia)

#reorganizamos columnas

data= data[['Country', 'CountryCode', 'Province', 'City', 'CityCode', 'Lat', 'Lon', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Date']]
data=data.iloc[:,[0,7,8,9,10,11]]


#cambiamos fecha a variable categorica
#2020-03-06
#2020-05-05

x= data.iloc[:,5:6]
y= data.iloc[:,1]

x=np.array(x)
y=np.array(y)

#categorizar la fecha
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder() #determinar la decodificacion para la variable x
x[:,0] =labelEncoder_X.fit_transform(x[:,0]) #en este caso no es optimo debido a que paises no son variables categoricas ordinales


#dividir el dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/5, random_state=0)

#entrenar el modelo de regesión lineal simple

from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(x_train, y_train)


#visualización de datos entrenamiento

plt.scatter(x_train, y_train, color="red") #nube de puntos
plt.plot(x_train, regresion.predict(x_train), color="blue")
plt.title("Casos positivos de covid")
plt.xlabel("dias")
plt.ylabel("Número de contagiados")
plt.show()   

#el crecimiento de infectados no presenta un comportamiento lineal

#regresion lineal polinómica

#ajuste del modelo de regresion lineal polinómica
#dividir de nuevo el dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=1/5, random_state=0)

lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)   


#graficar la regresion polinomica

plt.scatter(x,y, color="red")
plt.plot(x, lin_reg2.predict(x_poly), color="blue")
plt.title("Modelo de regresión polinomial para contagios" )
plt.xlabel("Dias")
plt.ylabel("Número de contagiados")
plt.show()    

#predicciones
y_predict_contagios =lin_reg2.predict(x_test)

#score de la predicción
lin_reg2.score(x_poly,y)

#calculo de MSE (mean square error)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_predict_contagios)


#calculo de desviacion estandar
pd.Series.std(pd.DataFrame(y))

#prediccion de contagios para dia 90,120,150,180
print("Día 62: ", lin_reg2.predict([[1,62,3844]]))
print("Día 63: ", lin_reg2.predict([[1,63,3969]]))
print("Día 64: ", lin_reg2.predict([[1,64,4096]]))
print("Día 65: ", lin_reg2.predict([[1,65,4225]]))
print("Día 66: ",lin_reg2.predict([[1,66, 4356]]))