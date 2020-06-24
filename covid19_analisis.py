# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:45:20 2020

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

columnas= list(data.columns.values)
columnas= ['Country', 'CountryCode', 'Province', 'City', 'CityCode', 'Lat', 'Lon', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Date']

data= data[['Country', 'CountryCode', 'Province', 'City', 'CityCode', 'Lat', 'Lon', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Date']]

data=data.iloc[:,[0,7,8,9,10,11]]


#miramos correlacion

data.corr(method="pearson")


#cambiamos fecha a variable categorica
#2020-03-06
#2020-05-05

x= data.iloc[:,5:6]
y= data.iloc[:,2]

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

#entrenar el modelo
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()
regresion.fit(x_train, y_train)



#predecir el conjunto de test
y_predict = regresion.predict(x_test)

#visualización de datos entrenamiento

plt.scatter(x_train, y_train, color="red") #nube de puntos
plt.plot(x_train, regresion.predict(x_train), color="blue")
plt.title("muertes covid (conjunto de train)")
plt.xlabel("dias")
plt.ylabel("Muertes")
plt.show()  


#regresion lineal polinómica

#ajuste del modelo de regresion lineal polinómica
#dividir de nuevo el dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=1/5, random_state=0)

lin_reg2= LinearRegression()
lin_reg2.fit(x_train,y_train)   


#prediccion

y_predict_polinomica= lin_reg2.predict(x_test)  

#graficar la regresion polinomica

plt.scatter(x,y, color="red")
plt.plot(x, lin_reg2.predict(x_poly), color="blue")
plt.title("Modelo de regresión polinomial" )
plt.xlabel("Dias")
plt.ylabel("Muertes por covid")
plt.show() 

#predicciones para 90 120 150 y 180 dias

print("Día 90: ", lin_reg2.predict([[1,90,8100]])) #981
print("Día 120: ", lin_reg2.predict([[1,120,14400]])) #1872
print("Día 150: ", lin_reg2.predict([[1,150,22500]])) #3048
print("Día 180: ",lin_reg2.predict([[1,180, 32400]])) #4508

print("Día 62: ", lin_reg2.predict([[1,62,3844]])) #405
print("Día 63: ", lin_reg2.predict([[1,63,3969]])) #421
print("Día 64: ", lin_reg2.predict([[1,64,4096]])) #437
print("Día 65: ", lin_reg2.predict([[1,65,4225]])) #454

print("Día 72: ", lin_reg2.predict([[1,72,5184]])) #582 muertos
 

#predicciones para dia 70 a 80
for i in range(70, 81):
    print("Dia ", i, lin_reg2.predict([[1,i,i**2]]))
    

"""
Dia  70 [544.28575282]
Dia  71 [563.11330609]
Dia  72 [582.25666567]
Dia  73 [601.71583156]
Dia  74 [621.49080378]
Dia  75 [641.58158231]
Dia  76 [661.98816715]
Dia  77 [682.71055832]
Dia  78 [703.7487558]
Dia  79 [725.10275959]
Dia  80 [746.7725697]
"""

#predicciones para 	dias 81 a 90

for i in range(81, 91):
    print("Dia ", i, lin_reg2.predict([[1,i,i**2]]))

"""
Dia  81 [768.75818613]
Dia  82 [791.05960888]
Dia  83 [813.67683794]
Dia  84 [836.60987332]
Dia  85 [859.85871501]
Dia  86 [883.42336302]
Dia  87 [907.30381735]
Dia  88 [931.50007799]
Dia  89 [956.01214495]
Dia  90 [980.84001822]
"""


