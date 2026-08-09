import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



fs = 100_000_000          # Sampling Frequency

duration = 0.002          # 2 ms

t = np.arange(0, duration, 1/fs)
#print(len(t))

noise = np.random.normal(0,0.02,len(t))
healthy = noise # HEALTHY SIGNAL 

#CORONA SIGNAL GENERATION//
corona = noise.copy()

pulse_positions = np.random.randint(0,len(t),70)

for p in pulse_positions:
    corona[p:p+8] += np.hanning(8)*0.25

#SURFACE DISCHARGE
surface = noise.copy()

pulse_positions = np.random.randint(0,len(t),45)

for p in pulse_positions:
    surface[p:p+12] += np.hanning(12)*0.55


#INTERNAL DISCHARGE
internal = noise.copy()

pulse_positions = np.random.randint(0,len(t),30)

for p in pulse_positions:
    internal[p:p+15] += np.hanning(15)*1.0

#CRACK 
crack = noise.copy()

pulse_positions = np.random.randint(0,len(t),20)

for p in pulse_positions:

    width = np.random.randint(8,20)

    amplitude = np.random.uniform(0.8,1.8)

    crack[p:p+width] += np.hanning(width)*amplitude   


# PRINTING THE GRAPHS
signals = {
    "Healthy": healthy,
    "Corona": corona,
    "Surface": surface,
    "Internal": internal,
    "Crack": crack
}

plt.figure(figsize=(15,12))

for i,(name,signal) in enumerate(signals.items()):

    plt.subplot(5,1,i+1)

    plt.plot(t*1000,signal)

    plt.title(name)

    plt.grid(True)

plt.tight_layout()

plt.show()    

