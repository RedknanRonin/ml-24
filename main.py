import matplotlib
import numpy as np
import pandas as pd
import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("mxmh_survey_results.csv")

df = pd.get_dummies(df, columns = ['Fav genre'], dtype=int)


df["While working"] = df["While working"].apply(lambda x: 1 if x=="Yes" else 0)
df["Instrumentalist"] = df["Instrumentalist"].apply(lambda x: 1 if x=="Yes" else 0)
df["Composer"] = df["Composer"].apply(lambda x: 1 if x=="Yes" else 0)
df["Foreign languages"] = df["Foreign languages"].apply(lambda x: 1 if x=="Yes" else 0)
df["Exploratory"] = df["Exploratory"].apply(lambda x: 1 if x=="Yes" else 0)

df = df.dropna()
df= df.select_dtypes(include=['number'])
data= df


def modelByLinearRegression(a) :

    anxiety = data[['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                       'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                       'Fav genre_Rock', 'Fav genre_Video game music', a]]

    ct = ColumnTransformer([
        ('somename', StandardScaler(), ['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                                        'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                                        'Fav genre_Rock', 'Fav genre_Video game music'])
    ], remainder='passthrough')
    b = ct.fit_transform(anxiety)
    X = b[:, :24]
    y = b[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.2)
    regr = LinearRegression().fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    print(f"\nR^2 score for Linear Regression in {a}: ",regr.score(X_test, y_test))
    print("Mean squared error for linear regression: ",mean_squared_error(y_test, y_pred))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(a)
    plt.show()


def modelByLassoRegression(cond):
    anxiety = data[['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                    'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                    'Fav genre_Rock', 'Fav genre_Video game music', cond]]

    ct = ColumnTransformer([
        ('somename', StandardScaler(), ['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                                        'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                                        'Fav genre_Rock', 'Fav genre_Video game music'])
    ], remainder='passthrough')
    b = ct.fit_transform(anxiety[['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                                  'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                                  'Fav genre_Rock', 'Fav genre_Video game music', cond]])
    X = b[:, :24]
    y = b[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.2)
    regr = Lasso(0.09).fit(X_train,y_train)

    y_pred = regr.predict(X_test)
    print(f"\nR^2 score for Lasso regression in {cond}: ",regr.score(X_test, y_test))
    print("Mean squared error for lasso regression: ",mean_squared_error(y_test, y_pred))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(cond)
    plt.show()

condition = ['Anxiety', 'OCD', 'Depression', 'Insomnia']

for i in condition:
    #modelByLinearRegression(i)
    modelByLassoRegression(i)



