import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.sin(np.arange(200)/(5*np.pi)))
klicker = clicker(ax, ["event"], markers=["x"])

plt.show()

print(klicker.get_positions())