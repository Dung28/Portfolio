#%% - Import
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
# cây quyết định
#%%- Configs
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=200
plt.rcParams['font.size']= 13

#%% Load data
df = pd.read_csv('./data/Buys_computer.csv')
x=df.drop("Buys_computer", axis='columns')
y=df['Buys_computer']

#%% -
from sklearn.preprocessing import LabelEncoder
x['Age_n'] = LabelEncoder().fit_transform(x['Age'])
# d= {'<=30:':0, '31..40':1, '>40':2}
# x['Age_n']=x['Age'].map(d)
x['Income_n'] = LabelEncoder().fit_transform(x['Income'])
x['Student_n'] = LabelEncoder().fit_transform(x['Student'])
x['Credit_rating_n'] = LabelEncoder().fit_transform(x['Credit_rating'])
x_n = x.drop(['Age', 'Income', 'Student', 'Credit_rating'], axis='columns')
y_n = LabelEncoder().fit_transform(y)

#%% -  Fit model (DL 14 dòng nên k tách)
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# Supported criteria are 'gini' for the Gini index and 'Entropy' for the information gain.
model = DecisionTreeClassifier(criterion='gini', random_state=10).fit(x_n, y_n)
# model = DecisionTreeClassifier(criterion='entropy', random_state=100).fit(x_n, y_n)

#%% - test score
score = model.score(x_n,y_n)

#%% - Visualize results
features = ['Age', 'Income', 'Student', 'Credit_rating']
text_representation = tree.export_text(model, feature_names=features)
print(text_representation)

#%%
plt.figure(figsize=(20,20), dpi=150)
t = tree.plot_tree(model, feature_names=features, class_names=['No', 'Yes'], filled=True)
plt.show()

#%% - Prediction,
# Age<30, Income: low, Student: yes, Credit: fair
buy_computer = model.predict([[1,1,1,1]])
print(buy_computer)


#%% - SALARIES
# #%% - Import
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# # from pmdarima.arima import auto_arima
# import warnings
# warnings.filterwarnings('ignore')
# # cây quyết định
# #%%- Configs
# plt.rcParams['figure.figsize']=(10,8)
# plt.rcParams['figure.dpi']=200
# plt.rcParams['font.size']= 13

#%% Load data
df = pd.read_csv('./data/Salaries.csv')
x=df.drop("Salary_more_than_100k", axis='columns')
y=df['Salary_more_than_100k']

#%%
from sklearn.preprocessing import LabelEncoder
x['Company_n'] = LabelEncoder().fit_transform(x['Company'])
x['Job_n'] = LabelEncoder().fit_transform(x['Job'])
x['Degree_n'] = LabelEncoder().fit_transform(x['Degree'])
x_n = x.drop(['Company', 'Job', 'Degree'], axis='columns')
y_n = LabelEncoder().fit_transform(y)

#%% -  Fit model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# Supported criteria are 'gini' for the Gini index and 'Entropy' for the information gain.
model = DecisionTreeClassifier(criterion='gini', random_state=10).fit(x_n, y_n)

#%% - test score
score = model.score(x_n,y_n)

#%% - Visualize results
features = ['Company', 'Job', 'Degree']
text_representation = tree.export_text(model, feature_names=features)
print(text_representation)

#%%
plt.figure(figsize=(20,20), dpi=150)
t = tree.plot_tree(model, feature_names=features, class_names=['No', 'Yes'], filled=True)
plt.show()