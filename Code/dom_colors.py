# In this program we find the 10 dominant colors in an image and then calculates its entropy
# Author Roshan Vemula

# Import Libraries
import matplotlib.image as img
from matplotlib import pyplot as plt
from scipy.cluster.vq import whiten, kmeans
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Read the image and print dimensions
image = img.imread('rainbow.jpeg')
print(image.shape)

# Store RGB values of all pixels in lists r, g and b
r = []
g = []
b = []
for row in image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

# Printing the sizes of lists r, g, b
print(len(r))
print(len(g))
print(len(b))

# Saving the r,g,b lists as pandas data frame
df = pd.DataFrame({'red': r, 'green': g, 'blue': b})

# Scaling the values before performing k-means by whiten method
df['scaled_color_red'] = whiten(df['red'])
df['scaled_color_blue'] = whiten(df['blue'])
df['scaled_color_green'] = whiten(df['green'])

# Perform k-means
cluster_centers, _ = kmeans(df[['scaled_color_red', 'scaled_color_blue', 'scaled_color_green']], 10)

dominant_colors = []

# Get standard deviations of each color
red_std, green_std, blue_std = df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    red_scaled, green_scaled, blue_scaled = cluster_center

# Convert each standardized value to scaled value
    dominant_colors.append((
        red_scaled * red_std / 255,
        green_scaled * green_std / 255,
        blue_scaled * blue_std / 255
    ))

# Display colors of cluster centers
plt.title("10 Dominant Colors")
plt.imshow([dominant_colors])
plt.show()

# Converting input image to numpy array
n_ary = np.array(image)
lists = n_ary.tolist()

# Converting array to list
pd_series = pd.Series(lists)
count = pd_series.value_counts()

# Calculating the Entropy
Entropy = entropy(count)
print(f'The entropy is {Entropy}')
