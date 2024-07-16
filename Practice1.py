import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Generate 10 random 3-channel images
images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(10)]

# Step 2: Convert images to grayscale
grayscale_images = [np.mean(img, axis=2) for img in images]

# Step 3: Calculate statistics for each image
data = []
for i, img in enumerate(grayscale_images):
    max_val = np.max(img)
    min_val = np.min(img)
    mean_val = np.mean(img)
    std_val = np.std(img)
    data.append([max_val, min_val, mean_val, std_val])

# Step 4: Save the statistics in an Excel file
df = pd.DataFrame(data, columns=['Max', 'Min', 'Mean', 'Std'])
output_path = "image_statistics.xlsx"
df.to_excel(output_path, index_label='Image')

print(f"Image statistics saved to {output_path}")

# Display the statistics DataFrame
print(df)

# Display three of the randomly generated images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, img in zip(axes, images[:3]):
    ax.imshow(img)
    ax.axis('off')

plt.show()

