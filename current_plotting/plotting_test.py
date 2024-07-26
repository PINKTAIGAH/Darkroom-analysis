import numpy as np
import matplotlib.pyplot as plt

FILEPATH = "/Users/giorgio/Gsi_data/Eris_run004.txt"

data = np.loadtxt(FILEPATH, dtype=str, usecols=range(6),)

header_mask = data == "time"
header_loc = np.argwhere(header_mask)[:, 0]
print(header_loc)

trimmed_data = np.delete(data, header_loc, axis=0).astype(np.float32)

# plt.plot(trimmed_data[:, 0], trimmed_data[:, 2], )
plt.plot(trimmed_data[:, 0], trimmed_data[:, 2])
plt.show()