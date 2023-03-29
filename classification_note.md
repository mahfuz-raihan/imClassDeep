## 1. Architecture of a classification neural network
|**Hyperparameter**| **Binary Classification**  | **Multiclass Classfication**  |
|:----------------:|----------------------------|-------------------------------|
|Input layer shape ```(in_features)```| Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction)  | As like binary classification  |
|**Hidden layer(s)**| Problem specific, minimum = 1, maximum = unlimited  | As like binary classification |
| **Neurons per hidden layer**  | Problem specific, generally 10 to 512  | As like binary classification  |
| **Output layer shape** ```(out_features)```  | 1 (one or other)  | 1 per class (e.g. 3 for food, person or dog photo)  |
| **Hidden layer activation**  | Usually ReLU (rectified linear unit) but can be many others  | As like binary classification |
| **Output activation**  | Sigmoid (```torch.sigmoid``` in PyTorch)  | Softmax (```torch.softmax``` in pytorch)  |
| **Loss fucntion**  | Binary Crossentropy (```torch.nn.BCELoss``` in pytorch)  | Cross Entroyp (```torch.nn.CrossEntropyLoss``` in pytorch) |
| Optimizer  | ```SGD``` (stochastic gradient descent), ```Adam``` optimizer  | as like binary classification  |

## 2. Data preparation and make it of suitable model
#### Turn data into tensors and create train and test splits
We've explored our data's output and input shapes; now let's prepare it for usage with PyTorch and modeling.

Specifically, we'll need to:

1. Convert our data into tensors (right now our data is in NumPy arrays and PyTorch prefers to work with PyTorch tensors).
2. Separate our data into training and test sets (we'll train a model on the training set to learn the relationships among "'X"' and "'y,"' and then test the discovered patterns on the test dataset).

**part 1:**
```python
# Turn data into tensors
# Otherwise this causes issues with computations later on
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples
print(X[:5], y[:5])
```
**part 2**
```python
# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(len(X_train), len(X_test), len(y_train), len(y_test))
```

## 3. How to build a model
We'll break it down into a few parts.
1. Setting up device agnostic code (so our model can run on CPU or GPU if it's available).
2. Constructing a model by subclassing ```nn.Module```.
3. Defining a loss function and optimizer.
4. Creating a training loop (this'll be in the next section).
> Note: check the notebook 'imClass'

### **Loss fucntion and optimizer**
There has some loss fucntion/cost function and optimizer for the model improvement
|  Loss function/Optimizer | Problem type  | Pytorch code  |
|:-:|---|---|
|   |   |   |
|   |   |   |
|   |   |   |
|   |   |   |
|   |   |   |
|   |   |   |