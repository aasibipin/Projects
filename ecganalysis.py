import scipy
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

ecgdata = "bio_signals/t1/Saved_dataECG.csv"
ecg = pd.read_csv(ecgdata, header=None, names= ["T1","Time", "Value"])
ecg = ecg.iloc[28:]

sample_size = 1000
sample_ecg = ecg.iloc[:sample_size,[1,2]]


start = sample_ecg.iloc[0,0]
time = sample_ecg.iloc[:,0] - start
value = sample_ecg.iloc[:,1]

plt.plot (time,value)
plt.show()
