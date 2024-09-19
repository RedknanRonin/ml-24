import numpy as np
import pandas as pd

df = pd.read_csv("mxmh_survey_results.csv")

data= df[["Age",'Hours per day','Fav genre','While working','Anxiety','Depression','Insomnia','OCD']]

# calculate average for mental health
data["Mental Health"] = (data["Anxiety"] + data["Depression"] + data["Insomnia"] + data["OCD"])/4
data.drop(columns=['Anxiety','Depression','Insomnia','OCD'],inplace=True)


data["While working"] = data["While working"].apply(lambda x: 1 if x=="Yes" else 0)  # change result to binary

print(data)