import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Minimum Wage Data.csv", encoding = "latin")
df.to_csv("Minwage.csv", encoding = "UTF-8")

df = pd.read_csv("Minwage.csv")

gb=df.groupby("State")

act_min_wage = pd.DataFrame()

for name, group in df.groupby("State"):
    if act_min_wage.empty:
        act_min_wage = group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name})
    else:
        act_min_wage = act_min_wage.join(group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name}))


issue_df = df[df["Low.2018"]==0]
issue_df["State"].unique()

min_wage_corr = act_min_wage.replace(0,np.NaN).dropna(axis=1).corr() #axis = 1 removes the column, axis = 0 removes the rows


'''
labels = [c[:2] for c in min_wage_corr.columns]


dfs= pd.read_html("https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes")

for df in dfs:
    print (df)


state_abbv = dfs[0] 

state_abbv.to_csv("State_Abbv.csv", index = False)
'''

state_abbv = pd.read_csv("State_Abbv.csv")
abbv_dict = state_abbv.set_index("State/District")["Postal Code"].to_dict()

#Hard coding missing states in dictionary

abbv_dict["Federal (FLSA)"] = "FLSA"
abbv_dict["Guam"] = "GU"
abbv_dict["Puerto Rico"] = "PR"

labels = [abbv_dict[c] for c in min_wage_corr.columns]
