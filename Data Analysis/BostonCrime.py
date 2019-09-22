import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import WMTSTileSource


df = pd.read_csv("crime.csv", encoding= "latin")
df.columns


legend = pd.read_csv("offense_codes.csv", encoding= 'latin')

legend_dict = legend.set_index("CODE")["NAME"].to_dict()

df.replace({"OFFENSE_CODE":legend_dict}, inplace  = True)   # OR use : df["OFFENSE_CODE"].replace(legend_dict, inplace = True)

df_drug = df[df["OFFENSE_CODE_GROUP"].str.contains("Drug")]

