import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("avocado.csv")

albany_df = df.copy()[df["region"] == "Albany"]

albany_df.set_index("Date",inplace = True)      #set_index only returns a dataframe, does not change the df
albany_df.sort_index(inplace = True)

albany_df['prices25ma'] = albany_df["AveragePrice"].rolling(25).mean()

df['region'].unique()

graph_df = 
