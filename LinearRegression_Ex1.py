#%% - Import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#%% - Some Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 16

#%% - load data
df = pd.read_csv('./data/Income.csv')
income = df[['Income']]
# df[['Income']]: kiểu dataframe
# df['Income']: kiểu
expenditure = df[["Expenditure"]]


#%% - Visualization
plt.scatter(income, expenditure)
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.show()

#%% - Create model
model = LinearRegression().fit(income, expenditure)

#%% -  Get results
intercept = model.intercept_
slope = model.coef_
R_square = model.score(income, expenditure)

#%% - Prediction
predicted_values = model.predict(income)
#%% - Future Prediction
future_values = np.array([26, 28, 31]).reshape(-1, 1)
f_predicted_values = intercept + slope * future_values
#c2: f_predicted_values = model.predict(future_values)

#%% - Visualization
plt.plot(income, expenditure, color='r', label='Actual')
plt.plot(income, predicted_values, color='g', linestyle='--', marker='x', label='Predict')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.show()
