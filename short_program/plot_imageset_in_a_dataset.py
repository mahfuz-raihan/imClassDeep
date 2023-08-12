"""
This program is a sample code using pytorch. you need to import pytorch as well.
another matplotlib is also needed to run this program
"""
# import libraries
import torch as t
import matplotlib.pyplot as plt
import torchvision as tv

# Import the datasets including train data and test data
# define 'class_name' using torch
train_data, test_date = "import dataset with different source or torchvision dataset" 
class_names = train_data.classes # this tis the folder name of vision dataset

# plot more image
t.manual_seed(42) # random fix value
fig = plt.figure(figsize=(9,9)) # define figure size in matplotlib
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = t.randint(0, len(train_data), size=[1]).item() # torch randit() library
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis(False);