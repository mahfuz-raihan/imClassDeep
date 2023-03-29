## Architecture of a classification neural network
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