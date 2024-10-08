import matplotlib
import numpy as np
import pandas as pd
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
df["Mental Health"] = (df["Anxiety"] + df["Depression"] + df["Insomnia"] + df["OCD"])/4
df.drop(["Anxiety", "Depression", "Insomnia", "OCD"], axis=1, inplace=True)
df = df.dropna()
df= df.select_dtypes(include=['number'])
data= df


def modelByLinearRegression():

    encoded = df.select_dtypes(include=['number'])

    a = encoded[['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                 'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                 'Fav genre_Rock', 'Fav genre_Video game music', 'Mental Health']]
    #print(a)

    ct = ColumnTransformer([
        ('somename', StandardScaler(), ['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                                        'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                                        'Fav genre_Rock', 'Fav genre_Video game music'])
    ], remainder='passthrough')
    b = ct.fit_transform(a[['Age','Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'BPM', 'Fav genre_Classical', 'Fav genre_Country', 'Fav genre_EDM', 'Fav genre_Folk',
                            'Fav genre_Gospel', 'Fav genre_Hip hop', 'Fav genre_Jazz', 'Fav genre_K pop', 'Fav genre_Latin', 'Fav genre_Lofi', 'Fav genre_Metal', 'Fav genre_Pop', 'Fav genre_R&B', 'Fav genre_Rap',
                            'Fav genre_Rock', 'Fav genre_Video game music', 'Mental Health']])
    #print(b)

    X = b[:, :24]
    y = b[:, -1]
    #print(X)
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20, test_size = 0.2)
    y_test.shape
    regr = LinearRegression().fit(X_train,y_train)
    y_pred = regr.predict(X_test)

    mae=mean_absolute_error(y_pred, y_test)
    mse=mean_squared_error(y_test, y_pred)
    print("Mean Absolute Error for linear regression: ", mae)
    print("Mean Squared Error for linear regression: ", mse)
    print("Training score for linear regression: ", regr.score(X_train, y_train))


    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Linear Regression with PCA Components')
    plt.show()


def modelByLassoRegression():
    #X contain 'Age','Hours per day', 'While working', and fav genre columns
    X = data.drop(["Mental Health"], axis=1)
    y = data["Mental Health"]
    #print(y.sum()/len(y))


    (X_train,X_test,y_train,y_test)=train_test_split(X,y)
    model= Lasso(1.8).fit(X_train,y_train)
    y_pred=model.predict(X_test)   # predicting values for

    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error for Lasso regression: ", mse)
    print("Training score for Lasso regression: ", model.score(X_train, y_train))

    plt.scatter(y_test, y_pred, color="b")
    plt.xlabel('Actual mental health score')    # set the label for the x/y-axis
    plt.ylabel('Predicted mental health score')

    plt.title("Actual mental health score in relation to predicted score",fontsize=14)

    plt.show()  # display the plot on the screen

modelByLinearRegression()
modelByLassoRegression()


