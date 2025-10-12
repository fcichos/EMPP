# %% Cell 1
#
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.spatial.distance import cdist


fig, ax = plt.subplots(figsize=(9,9))

n_steps=200

for _ in range(n_steps):
    clear_output(wait=True)


    display(fig)
    plt.pause(0.01)
    ax.clear()
