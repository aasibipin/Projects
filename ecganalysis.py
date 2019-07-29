import scipy
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

ecgdata = "bio_signals/t1/Saved_dataECG.csv"
ecg = pd.read_csv(ecgdata, header=None, names= ["Time","IDK", "Value"])

sample_size = 1000
sample_ecg = ecg.iloc[30:sample_size,[0,2]]

time = sample_ecg.iloc[:,0]
value = sample_ecg.iloc[:,1]

plt.plot (np.arange(time.size),value)
plt.show()
