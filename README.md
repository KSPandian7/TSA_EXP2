### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
### Date:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

## AIM:
To implement Linear and Polynomial Trend Estimation using Python.

## ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib, Pandas).
2. Load the dataset.
3. Calculate the linear trend values using the least square method.
4. Calculate the polynomial trend values using the least square method.
5. Display the graphs and print the trend equations.

## PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv("/content/tsla_2014_2023 (1).csv")
data.head()
data['date']=pd.to_datetime(data['date'])
data.info()
X=np.arange(len(data)).reshape(-1,1)
y=data['close'].values
```
### A - LINEAR TREND ESTIMATION
```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_linear_predict=regressor.predict(X)

# Graph for Linear trend
plt.figure(figsize=(10, 6))
plt.plot(data['date'], y, label='Actual Data', color='black')
plt.plot(data['date'], y_linear_predict, label='Linear Trend', color='blue')
plt.title('Tsla Stock predication')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# print the Linear Trend Equation
print(f"Linear Trend Equation: y = {regressor.coef_[0]:.2f} * x + {regressor.intercept_:.2f}")

```
### B- POLYNOMIAL TREND ESTIMATION
```python
from sklearn.preprocessing import PolynomialFeatures
# polynomial trend for degree 2
degree=2
poly_reg=PolynomialFeatures(degree=degree)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
regressor_poly=LinearRegression()
regressor_poly.fit(X_poly,y)
y_predict_poly_2=regressor_poly.predict(X_poly)

# polynomial trend for degree 3
degree_3=3
poly_reg_3=PolynomialFeatures(degree=degree_3)  # Use degree_3 here
X_poly_3=poly_reg_3.fit_transform(X)
poly_reg_3.fit(X_poly_3,y)
regressor_poly_3=LinearRegression()
regressor_poly_3.fit(X_poly_3,y)
y_predict_poly_3=regressor_poly_3.predict(X_poly_3)

# Graph for polynomial trend
plt.figure(figsize=(10, 6))
plt.plot(data['date'], y, label='Actual Data', color='black')
plt.plot(data['date'], y_predict_poly_2, label=f'Polynomial Trend (Degree {degree})', linestyle='-.', color='green')
plt.plot(data['date'], y_predict_poly_3, label=f'Polynomial Trend (Degree {degree_3})',  color='red')
plt.title('Tsla Stock predication')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# print the Polynomial Equations
print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly.coef_[2], regressor_poly.coef_[1], regressor_poly.intercept_))
print("Polynomial Trend Equation (Degree 3): y = {:.2f} * x^3 + {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly_3.coef_[3], regressor_poly_3.coef_[2], regressor_poly_3.coef_[1], regressor_poly_3.intercept_))
```
## OUTPUT
#### A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/8da9cfd6-d918-4ff8-888a-5fa0b8e7d6b2)
```
Linear Trend Equation: y = 0.06 * x + -631.58
```
#### B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/ac96f6c4-ed81-49b0-9c5e-8b849891f6b0)
```
Polynomial Trend Equation (Degree 2): y = 0.00 * x^2 + 0.03 * x + -283.25

Polynomial Trend Equation (Degree 3): y = -0.00 * x^3 + 0.00 * x^2 + -0.12 * x + 746.35
```

### RESULT:
Thus the python program for linear and Polynomial Trend Estimation has been executed successfully.
