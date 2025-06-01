

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')
x = np.arange(-0.5, 0.5, 0.00001)
y = np.tanh(2* np.pi * x)
plt.plot(y, linewidth=10)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
# plt.show()
plt.savefig("sigmoid.svg")
