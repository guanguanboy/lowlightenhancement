import matplotlib

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('pred.png')


plt.imshow(img)
plt.show()

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
plt.plot(x, y_sin)
plt.show()