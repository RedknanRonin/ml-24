import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as snsimport
matplotlib as plt

df = pd.read_csv("mxmh_survey_results.csv")

data= df[["Age",'Hours per day','Fav genre','While working','Anxiety','Depression','Insomnia','OCD']]

# calculate average for mental health
data["Mental Health"] = (data["Anxiety"] + data["Depression"] + data["Insomnia"] + data["OCD"])/4
data.drop(columns=['Anxiety','Depression','Insomnia','OCD'],inplace=True)


data["While working"] = data["While working"].apply(lambda x: 1 if x=="Yes" else 0)  # change result to binary


df["Instrumentalist"] = df["Instrumentalist"].apply(lambda x: 1 if x=="Yes" else 0)
df["Composer"] = df["Composer"].apply(lambda x: 1 if x=="Yes" else 0)
df["Foreign languages"] = df["Foreign languages"].apply(lambda x: 1 if x=="Yes" else 0)
df["Exploratory"] = df["Exploratory"].apply(lambda x: 1 if x=="Yes" else 0)
numeric_df = df.select_dtypes(include='number')
numeric_df["Mental Health"] = (df["Anxiety"] + df["Depression"] + df["Insomnia"] + df["OCD"])/4
print(numeric_df)

sns.heatmap(numeric_df.corr(), annot = True, fmt='.2f', annot_kws={"size": 8})

print(data)
(x_train,y_train)=train_test_split(data)
print(len(x_train),len(y_train))