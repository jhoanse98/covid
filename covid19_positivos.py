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


x= x[0:60,0:1] #60 dias desde el brote

y= y[0:60] #60 dias desde el caso 1

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
poly_reg= PolynomialFeatures(degree = 4)
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



for i in range(70, 81):
    print("Dia ", i, lin_reg2.predict([[1,i,i**2, i**3, i**4]]))
    

"""
Dia  70 [11264.50597141]
Dia  71 [11623.70902769]
Dia  72 [11988.54283235]
Dia  73 [12359.00738538]
Dia  74 [12735.10268679]
Dia  75 [13116.82873657]
Dia  76 [13504.18553473]
Dia  77 [13897.17308126]
Dia  78 [14295.79137617]
Dia  79 [14700.04041946]
Dia  80 [15109.92021112]
"""

#predicciones para dias 81 a 90

for i in range(81, 91):
    print("Dia ", i, lin_reg2.predict([[1,i,i**2,i**3, i**4]]))

""" Resultados anteriores con 
funcion polinómica de grado 2

Dia  81 [15525.43075116]  
Dia  82 [15946.57203957]  
Dia  83 [16373.34407636]
Dia  84 [16805.74686152]
Dia  85 [17243.78039506]
Dia  86 [17687.44467698]
Dia  87 [18136.73970727]
Dia  88 [18591.66548593]
Dia  89 [19052.22201297]
Dia  90 [19518.40928839]
"""

""" Funcion polinómica de grado 3
Dia  81 [22574.37560256]
Dia  82 [23479.4327471]
Dia  83 [24409.93926125]
Dia  84 [25366.26729055]
Dia  85 [26348.78898052]
Dia  86 [27357.87647669]
Dia  87 [28393.90192461]
Dia  88 [29457.23746979]
Dia  89 [30548.25525778]
Dia  90 [31667.3274341]

"""


"""Función polinómica de grado 4

Dia  81 [23173.99419563]
Dia  82 [24225.29977757]
Dia  83 [25317.91299342]
Dia  84 [26453.08700339]
Dia  85 [27632.09569745]
Dia  86 [28856.23369535]
Dia  87 [30126.81634655]
Dia  88 [31445.17973031]
Dia  89 [32812.68065563]
Dia  90 [34230.69666127]

"""

for i in range(91, 100):
    print("Dia ", i, lin_reg2.predict([[1,i,i**2,i**3, i**4]]))


""" Funcion polinomica de grado 3
Dia  91 [32814.82614429]
Dia  92 [33991.12353388]
Dia  93 [35196.59174839]
Dia  94 [36431.60293337]
Dia  95 [37696.52923434]
Dia  96 [38991.74279684]
Dia  97 [40317.6157664]
Dia  98 [41674.52028854]
Dia  99 [43062.82850881]
"""

"""Funcion polinómica de grado 4
Dia  91 [35700.62601573]
Dia  92 [37223.88771731]
Dia  93 [38801.92149402]
Dia  94 [40436.18780365]
Dia  95 [42128.16783376]
Dia  96 [43879.36350163]
Dia  97 [45691.29745433]
Dia  98 [47565.51306868]
Dia  99 [49503.57445126]
"""