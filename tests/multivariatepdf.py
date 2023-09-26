import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(0, 30, 300)
y = multivariate_normal.pdf(x, mean=0.0, cov=5.0)
fig, ax = plt.subplots()
ax.plot(x,y)
#ax.set_title("Probability Density Function")
ax.set_xlabel("Absolute distance [m]")
ax.set_ylabel("Probability density function")
plt.show()
