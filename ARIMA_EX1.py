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
#%% - Thiết lập thông số
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 16

#%% - Nạp Dl phân tích
df = pd.read_csv('./data/ACG.csv', index_col=0, parse_dates=True)
df.info()

#%% -  Vẽ biểu đồ giá đóng cửa cổ phiếu
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.show()

#%% - Chia tập DL
df_close = np.log(df['Close'])
train_data, test_data = df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close) * 0.9):]
plt.plot(train_data, 'Blue', label= 'Train data')
plt.plot(test_data, 'Red', label= 'Test data')
plt.ylabel('Close prices')
plt.xlabel('Date')
plt.legend()
plt.show()

#%% - Phân rã chuỗi dữ liệu
# Biểu đồ lịch sử so sánh giá đóng cửa với giá trị trung bình và độ lệch chuẩn của 12 kỳ trước đó
# Lấy 12 kỳ vì xem xét mối liên hệ trong quá khứ (lấy abo nhiêu cũng được => để xem sự tương quan thoi)
rolmean = train_data.rolling(12).mean()
rolstd = train_data.rolling(12).std()
plt.plot(train_data, color='blue', label='Original')
plt.plot(rolmean, color='r', label='Rolling mean')
plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend()
plt.show()

# Biểu đồ phân rã chuỗi thời gian (decompose)
decompose_results = seasonal_decompose(train_data, model='nultriplicative', period=30)
decompose_results.plot()
plt.show()

#%% - Kiểu định tính dừng của dữ liệu (Stationary)
def adf_test(data):
    indices = ['ADF: Test statistic', 'p value',"# of Lag", "# of Observations"]
    test = adfuller(train_data, autolag='AIC')
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f'Critical Value ({key})'] = value
    return results
print(adf_test(train_data))
print('----'*5)

def kpss_test(data):
    indices = ['KPSS: Test statistic', 'p value', "# of Lags"]
    test = kpss(train_data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f'Critical Value ({key})'] = value
    return results
print(kpss_test(train_data))

#%% - Kiểm định sự tương quan ( )
pd.plotting.lag_plot(train_data)
plt.show()

#%% - tương quan riêng
plot_pacf(train_data)
plt.show()

#%% -
plot_acf(train_data)
plt.show()

#%% - Chuyển đổi chuỗi dừng
# Tính sai phân bậc 1 dữ liệu train
diff = train_data.diff(1).dropna()
# diff = train_data.diff(1)

# Biểu đồ thể hiện dữ liệu ban đầu và sau khi lấy sai phân
fig, ax = plt.subplots(2, sharex='all')
train_data.plot(ax=ax[0], title="Giá đóng cửa")
diff.plot(ax=ax[1], title='Sai phân bậc nhất')
plt.show()

#%% - Kiểm tra lại tính dứng của dữ liệu sau khi tính sai phân
print(adf_test(diff))
print(kpss_test(diff))
plot_pacf(diff) # có thể xác định tham số 'p' cho mô hình ARINA
plt.show()

#%%
plot_acf(diff) #có thể xác định tham số 'q' cho mô hình ARINA
plt.show()

#%% - Xác định tham số p, q, d cho mô hình ARINA
# stepwise_fit = auto_arima(train_data, trace = True,)

# --------------------------------------------------------------
#%% - Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

#%% - Some Config
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 16

#%% - Load data
df = pd.read_csv("./data/ACG.csv", index_col=0, parse_dates= True)
df.info()

#%% - Visualization (Close)
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.show()

#%% - Chia tập dữ liệu
df_close = np.log(df['Close'])
train_data, test_data = df_close[:int(len(df_close) * 0.9)], df_close[int(len(df_close)*0.9):]
plt.plot(train_data, 'blue', label='Train data')
plt.plot(test_data, 'r', label='Test data')
plt.xlabel('Date')
plt.ylabel('Close prices')
plt.legend()
plt.show()

#%% - Phân rã chuỗi dữ liệu
# Biểu đồ lịch sử so sánh giá đóng cửa với giá trị trung bình và độ lệch chuẩn của những kỳ trước đó
rolmean = train_data.rolling(12).mean()
rolstd = train_data.rolling(12).std()
plt.plot(train_data, 'g', label='Original')
plt.plot(rolmean, 'r', label='Rolling mean')
plt.plot(rolstd, 'black', label='Rolling std')
plt.legend()
plt.show()

#Biểu đồ phân rã chuỗi thời gian (Decompose)
decompose_results = seasonal_decompose(train_data, model= 'multiplicative', period=30)
decompose_results.plot()
plt.show()

#%% - Kiểm định tính dừng của dữ liệu (Stationary)
def adf_test(data):
    indices = ['ADF: Test statistic', 'p value', '# of Lags', '# of Observations']
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f'Critical value ({key})'] = value
    return results

def kpss_test(data):
    indices = ['KPSS: Test statistic', 'p value', '# of Lags']
    kpss_test = kpss(data)
    results = pd.Series(kpss_test[:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f'Critical value ({key})'] = value
    return results
print(adf_test(train_data))
print("----"*5)
print(kpss_test(train_data))

#%% - Kiểm định tự tương quan (Auto Correlation)
pd.plotting.lag_plot(train_data)
plt.show()
#%%
plot_pacf(train_data)
plt.show()
#%%
plot_acf(train_data)
plt.show()

#%% - Chuyển dữ liệu --> chuỗi dừng
# Tính sai phân
diff = train_data.diff(1).dropna()
# Biểu đồ thể hiện dữ liệu ban đầu và sau khi lấy sai phân
fig, ax = plt.subplots(2, sharex='all')
train_data.plot(ax=ax[0], title='Giá đóng cửa')
diff.plot(ax= ax[1], title='Sai phân bậc nhất')
plt.show()

#%% - Kiểm tra lại tính dừng của dữ liệu sau khi lấy sai phân bậc 1
print(adf_test(diff))
print("------------"*8)
print(kpss_test(diff))
#%%
plot_pacf(diff) # --> xác định tham số "p" cho mô hình ARIMA
plt.show()
#%%
plot_acf(diff) # --> xác định tham số "q" cho mô hình ARIMA
plt.show()

    #%% - Xác định tham số p, d, q cho mô hình ARIMA
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary())
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%% - Tạo model
# model = ARIMA(train_data, order=(1,1,2))
# fitted = model.fit()
model = sm.tsa.arima.ARIMA(train_data, order=(1,1,2))
fitted = model.fit()

print(fitted.summary())

#%% - Dự báo (Forecast)
# fc = fitted.forecast(len(test_data), alpha=0.05)
# fc_series = pd.Series(fc, index=test_data.index)
# lower_series = pd.Series(conf[:, 0], index=test_data.index)
# upper_series = pd.Series(conf[:, 1], index=test_data.index)
# plt.figure(figsize=(16,10), dpi=150)
# plt.plot(train_data, 'orange', label="Training data")
# plt.plot(test_data, 'b', label='Actual stock price')
# plt.plot(fc_series, 'r', label='Predict stock price')
# plt.show()
# print(fc)
# print(fc_series)
gfc = fitted.get_forecast(len(test_data), alpha=0.05)
fc = gfc.predicted_mean
fc.index = test_data.index
se, se.index = gfc.se_mean, test_data.index
conf, conf.index = gfc.conf_int(alpha=0.05), test_data.index
conf = conf.to_numpy()
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(16,10), dpi=150)
plt.plot(train_data, 'b', label="Training data")
plt.plot(test_data, 'orange', label='Actual stock price')
plt.plot(fc, 'r', label='Predict stock price')
# plt.fill_between(upper_series.index, lower_series, upper_series, 'b', alpha=.10)
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
# print(fc)
# print(fc)

# lower_series = pd.Series(conf[:, 0], index=test_data.index)
# upper_series = pd.Series(conf[:, 1], index=test_data.index)
# conf = gfc.conf_int(alpha=0.05)
# lower_series = conf[:, 0], index=test_data.index
# upper_series = conf[:, 1], index=test_data.index
# plt.figure(figsize=(16,10), dpi=150)
# plt.plot(train_data, 'orange', label="Training data")
# plt.plot(test_data, 'b', label='Actual stock price')
# plt.plot(fc_series, 'r', label='Predict stock price')
# plt.show()
# print(gfc)
#------------------------------------------------------------

# cây quyết định
#%%- Configs
plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['figure.dpi']=200
plt.rcParams['font.size']= 13

#%% Load data
df = pd.read_csv('./data/Buys_computer.csv')
x=df.drop("Buys_computer", axis='columns')
y=df['Buys_computer']
