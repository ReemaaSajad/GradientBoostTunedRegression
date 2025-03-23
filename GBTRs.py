import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from numpy import nan
from numpy import absolute

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline

df = pd.read_csv("insurance.csv")
print(df.shape)
print(df.head())

print(df.info(), '\n')
print(df.describe(), '\n')
print(df.isnull().sum())

#Exploratory Data Analysis
#In [4]:
region_count = pd.DataFrame(df.region.value_counts())
region_charge_sum = pd.DataFrame(df.groupby('region').sum()['charges'])
region_charge_mean = pd.DataFrame(df.groupby('region').mean()['charges'])

fig, ax = plt.subplots(1, 2, figsize = (13, 5))

ax0 = sns.barplot(data = region_count, x = region_count.index.to_list(), 
                     y = region_count['region'], ax = ax[0])
for i in ax0.patches:
    ax0.text(x = i.get_x() + 0.2, y = i.get_height(), s = i.get_height(), fontsize = 14)
ax0.set_title("Region Count", fontsize = 12)   
    
ax1 = sns.barplot(data = region_charge_sum, x = region_charge_sum.index.to_list(), 
                  y = region_charge_sum['charges'], ax = ax[1], alpha = 0.5)  
ax2 = ax[1].twinx()
ax1a = sns.lineplot(data = region_charge_mean, x = region_charge_mean.index.to_list(), 
                    y = region_charge_mean['charges'], marker = 'o', markersize = 8, 
                    color = 'purple', ax = ax2)  
for i in ax1.patches:
    ax1.text(x = i.get_x() + 0.07, y = i.get_height() +1e5, s = "{:.3g}".format(i.get_height()), 
             fontsize = 14, color = "red")
    
ax1.set_title("Region Sum (Bar) and Region Mean (Line)", fontsize = 12)

for tick_label in ax1.axes.get_yticklabels():
    tick_label.set_color("red")
    tick_label.set_fontsize("12")

for tick_label in ax1a.axes.get_yticklabels():
    tick_label.set_color("purple")
    tick_label.set_fontsize("12")

#In [5]:
plt.figure(figsize = (11, 8))
sns.scatterplot(data = df, x = df.age, y = df.charges, hue = 'smoker', 
                style = 'sex', size = 'bmi', sizes = (20, 200), legend='auto')
plt.legend(loc = 'upper right', bbox_to_anchor = (1.13, 1))
plt.title("Charges vs Age, Smoker, bmi, sex", fontsize = 12)
plt.xlabel("Age", fontsize = 12); plt.ylabel("Charges", fontsize = 12)
plt.show()


#Observations :
#1.	Charges increas as age increases
#2.	Smokers cost much more than non-smokers
#3.	Smokers with high bmi cost more (almost double the charges)
#4.	Smokers with low bmi cost less than non-smokers with high bmi
#In [6]:
fig = px.box(df, x = 'children', y = 'charges', color = 'sex')
fig.update_layout(title = "Does the number of children affect the amount of charges for both genders?", 
                  paper_bgcolor = 'rgb(243, 243, 243)', 
                 plot_bgcolor = 'rgb(243, 243, 243)')

#012345010k20k30k40k50k60k
#sexfemalemaleDoes the number of children affect the amount of charges for both genders?childrencharges
#In [7]:
px.box(df, x = 'region', y = 'charges', color = 'children')
#southwestnorthwestsoutheastnortheast010k20k30k40k50k60k
#children013254regioncharges
#In [8]:
fig = px.pie(df, values = "charges", names = "children")
fig.update_layout(title = "How much does each section of the number of children account for the charges", 
                  paper_bgcolor = 'rgb(243, 243, 243)', 
                 plot_bgcolor = 'rgb(243, 243, 243)')
#40%23.2%20.4%13.6%1.95%0.891%
#012345How much does each section of the number of children account for the charges
#In [9]:
northeast = df.groupby(['region', 'children']).sum()['charges']['northeast']
northwest = df.groupby(['region', 'children']).sum()['charges']['northwest']
southeast = df.groupby(['region', 'children']).sum()['charges']['southeast']
southwest = df.groupby(['region', 'children']).sum()['charges']['southwest']

region_list = [northeast, northwest, southeast, southwest]
region_name_list = ['northeast', 'northwest', 'southeast', 'southwest']
plt.figure(figsize = (16,15), facecolor='white')
for num, (reg, name) in enumerate(zip(region_list, region_name_list)):
    plt.subplot(2, 2, num+1)
    labels = reg.values
    reg.plot.pie(autopct = "%.3g %%", pctdistance = 0.8, fontsize =16)
    plt.title(name, fontsize =16)
    plt.ylabel("")
plt.suptitle("How much does the number of Children account for the total charges in each region", 
             fontsize = 16)
plt.show()

#Distribution of numeric data
#In [10]:
numeric_data = df.select_dtypes(np.number)
fig, ax = plt.subplots(2, 2, figsize = (10, 6), constrained_layout = True)
ax = ax.flatten()
sns.set_style('darkgrid')
for num, col in enumerate(numeric_data.columns):
    sns.distplot(numeric_data[col], ax = ax[num])
plt.suptitle('Distribution of Numeric Data')
plt.show()

#In [11]:
cat_data = df[['sex', 'smoker', 'region', 'charges']]
fig, ax = plt.subplots(3, 1, figsize =(10, 16))
ax = ax.flatten()
for num, col in zip(range(3), cat_data.columns):
    sns.boxplot(data = cat_data, x = cat_data[col], y = cat_data['charges'], ax = ax[num])

#Build a Model
#Used MinMaxScaler to scale both the input features and the target.
#The reason we also scale the target is that it is much eaiser to determine if the values of Root Mean Square Error (RMSE), Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score are large or not. For example, if the RMSE is larger than 1, it means your model perform worse than a naive prediction.
#In [12]:

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X[self.attribute_names]

num_pipeline = Pipeline([
    ('select_numeric', DataFrameSelector(['age', 'bmi', 'children'])),
    ('minmax_scaler', MinMaxScaler())
])

cat_pipeline = Pipeline([
    ('select_cat', DataFrameSelector(['sex', 'smoker', 'region'])),
    ('one_hot', OneHotEncoder())
])

preprocessing_pipeline = FeatureUnion(transformer_list = [
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

y_pipeline = Pipeline([
    ('select_numeric', DataFrameSelector(['charges'])),
    ('minmax_scaler', MinMaxScaler())
])

X = preprocessing_pipeline.fit_transform(df).toarray()
y = y_pipeline.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =

 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# We are going to try and compare 12 different models

lin_reg = LinearRegression()
lasso = Lasso(random_state = 42)
ridge = Ridge(random_state = 42)
elastic_net = ElasticNet(random_state = 42)
sgd_reg = SGDRegressor(random_state = 42)
rand_reg = RandomForestRegressor(random_state = 42)
tree_reg = DecisionTreeRegressor(random_state = 42)
gb_boost = GradientBoostingRegressor(random_state = 42)
ada_boost = AdaBoostRegressor(random_state = 42)
knn_reg = KNeighborsRegressor()
svm = SVR(kernel='linear')
xgb_reg = XGBRegressor(random_state = 42)

regressor_list = [lin_reg, lasso, ridge, elastic_net, sgd_reg, rand_reg, 
                  tree_reg, gb_boost, ada_boost, knn_reg, svm, xgb_reg] 
regressor_name_list = ["lin_reg", "lasso", "ridge", "elastic_net", "sgd_reg", "rand_reg", 
                  "tree_reg", "gb_boost", "ada_boost", "knn_reg", "svm", "xgb_reg"] 
#In [14]:
rmse = []
mse = []
mae = []
r2 = []
y_predicted = []
for reg in regressor_list:
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_predicted.append(y_pred)
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mse.append(mean_squared_error(y_test, y_pred))
    mae.append(mean_absolute_error(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))
#Visualize the predictions
#In [15]:

fig, ax = plt.subplots(4, 3, sharex = True, sharey = True, figsize = (15,13))
models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Elastic Net', 'SGD Regressor',
         'RandomForest Regressor', 'DecisionTree Regressor', 'GradientBoost Regression', 
          'AdaBoost Regressor', 'KNN Regressor', 'SVM', 'XGBoost Regressor']
y_pred_models = y_predicted
ax = ax.flatten()
for num, (pred, model) in enumerate(zip(y_pred_models, models)):
    ax[num].scatter(pred, y_test, s=20)
    ax[num].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax[num].set_title(model, fontsize = 14)
    
fig.supxlabel('Predicted Values', fontsize = 14)
fig.supylabel('True Values', fontsize = 14)
plt.suptitle("True Values vs Predicted Values", fontsize = 14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#From the above graph, it seems like Gradient Boosting Regressor has the best results. Let's vertify that below.

compare_regressor = pd.DataFrame(regressor_name_list, columns = ['Model'])
compare_regressor['rmse'] = rmse
compare_regressor['mse'] = mse
compare_regressor['mae'] = mae
compare_regressor['r2'] = r2
print(compare_regressor.sort_values(by = 'rmse', ascending = True))


#Out[16]:
#Indeed, Gradient Boosting Regressor gives the best result (i.e. has the less error). Its RMSE is close to zero, it means the model performs well if there is no overfitting. Therefore, the next thing to do is to check if overfitting occurs and fix overfitting if it happens.
#Plot the learning curves to check if there overfitting occurs
#In [17]:
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X_train, X_test, y_train, y_test):
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        test_errors.append(mean_squared_error(y_test, y_test_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="test")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)
    plt.show()                
#In [18]:
plt.figure(figsize = (15, 10))
plot_learning_curves(gb_boost, X_train, X_test, y_train, y_test)
#It's overfitting. The reason is that the test set has a much higher RMSE values than the training set. Another quick way to determine is that there is a big gap between the training set and the test set.


#We notice that the gap is getting closer. This means that the model may have better performance if we feed more data to the model. However, we don't have any more data. Therefore, we need to tune the hyperparameters of Gradient Boosting Regressor to avoid overfitting from happening.
#The hyperparameters that I am going to tune:
#1.	Depth of each tree (max_depth)
#2.	Number of trees (n_estimators)
#3.	Learning rate (learning_rate)
#4.	Sub sample (subsample)
#Tune the Gradient Boosting Regressor Model
#1. Depth of each tree (max_depth)
#In [19]:
max_depths = np.arange(1, 10, 1)
param_range = max_depths

train_results = []; test_results = []
for max_depth in max_depths:
    model = GradientBoostingRegressor(max_depth = max_depth, random_state = 42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_results.append(train_rmse)
    
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_results.append(test_rmse)

plt.figure(figsize = (15, 8))
plt.plot(param_range, train_results, 'b-', label = "train")
plt.plot(param_range, test_results, 'r-', label = "test")
plt.legend(loc = 'upper right', fontsize = 16)
plt.xticks(param_range, rotation = 90, fontsize = 16)
plt.ylabel("RMSE", fontsize = 16)
plt.xlabel("Depth of each tree (max_depths)", fontsize = 16)
plt.show()

#Let's choose the depth of each tree (max_depth) as 2
#2. Number of trees (n_estimators)
#In [20]:
n_estimators = np.arange(10, 100, 5)
param_range = n_estimators

train_results = []; test_results = []
for n_estimator in n_estimators:
    model = GradientBoostingRegressor(n_estimators = n_estimator, random_state = 42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_results.append(train_rmse)
    
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_results.append(test_rmse)

plt.figure(figsize = (15, 8))
plt.plot(param_range, train_results, 'b-', label = "train")
plt.plot(param_range, test_results, 'r-', label = "test")
plt.legend(loc = 'upper right', fontsize = 16)
plt.xticks(param_range, rotation = 90, fontsize = 16)
plt.ylabel("RMSE", fontsize = 16)
plt.xlabel("Number of trees (n_estimators)", fontsize = 16)
plt.show()

#Let's choose the Number of trees (n_estimators) as 20
#3. Learning Rate
#In [21]:
learning_rates = np.arange(0.01, 0.05, 0.001)

train_results = []; test_results = []
for eta in learning_rates:
    model = GradientBoostingRegressor(learning_rate = eta, random_state = 42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_results.append(train_rmse)
    
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_results.append(test_rmse)

plt.figure(figsize = (15, 8))
plt.plot(learning_rates, train_results, 'b-', label = "train")
plt.plot(learning_rates, test_results, 'r-', label = "test")
plt.legend(loc = 'upper right', fontsize = 16)
plt.xticks(learning_rates, rotation = 90, fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylabel("RMSE", fontsize = 16)
plt.xlabel("Learning Rate", fontsize = 16)
plt.show()

#Let's choose the learning rate of 0.02
#4. Subsample
#In [22]:
subsamples = np.arange(0.01, 0.2, 0.01)
param_range = subsamples

train_results = []; test_results = []
for subsample in subsamples:
    model = GradientBoostingRegressor(subsample = subsample, random_state = 42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_results.append(train_rmse)
    
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_results.append(test_rmse)

plt.figure(figsize = (15, 8))
plt.plot(param_range, train_results, 'b-', label = "train")
plt.plot(param_range, test_results, 'r-', label = "test")
plt.legend(loc = 'upper right', fontsize = 16)
plt.xticks(param_range, rotation = 90, fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylabel("RMSE", fontsize = 16)
plt.xlabel("Subsample", fontsize = 16)
plt.show()

#Let's choose subsample = 0.03
#Hyperparameter Tunning Summary:
#1.	Depth of each tree (max_depth): 2
#2.	Number of trees (n_estimators): 20
#3.	Learning rate (learning_rate, eta): 0.02


#Train and predict again after tunning the hyperparameters
#In [23]:
gb_reg = GradientBoostingRegressor(max_depth = 2, n_estimators = 20, learning_rate = 0.02, 
                          subsample = 0.03, random_state = 42)
gb_reg.fit(X_train, y_train)
gb_reg_y_pred = gb_boost.predict(X_test)
gb_reg_rmse = np.sqrt(mean_squared_error(y_test, gb_reg_y_pred))
print("rmse: ", gb_reg_rmse)

plt.scatter(gb_reg_y_pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw = 2)
plt.show()
#rmse:  0.06893820445992102

#In [24]:
plt.figure(figsize = (15, 10))
plot_learning_curves(gb_reg, X_train, X_test, y_train, y_test)

#As we can see, two lines are close to each other. This means that there is no overfitting anymore.
#In [25]:
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Regressor After Tuning")
print("rmse: {}".format(rmse))
print("mse: {}".format(mse))
print("mae: {}".format(mae))
print("r2: {}".format(r2))
