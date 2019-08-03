import matplotlib.pyplot as mp
import numpy as np

x = np.linspace(-10, 10, 1000)
y = 1 /  (1+np.exp(-x))
mp.plot(x, y)
mp.show()

