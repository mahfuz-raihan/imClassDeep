# imClassDeep
This will use for pytorch note to determine the functionality of the pytorch and inner function

#### 3 step to making predictions with pytorch model:
1. Set the model with evaluation mode (```model.eval()```)
2. Make the predictions using inference mode context manager (```with torch.inference_mode():....```)
3. All prediction should be made with objects on the same device (weather data and model on GPU or CPU only)

### How to save and load a pytorch model? 
First we need to save a pytorch model, after that we can load the model for further work. Let's see how to save a pytorch model
The method saving and loading model in pytorch:
|PyTroch method|What does it do?|
|:------------:|:----------------|
|```torch.save```|Saves a seralized object to disk using python's pickle utility. Models, tensors and various other python object like dictionaries can be saved using ```torch.save```.|
|```torch.load```|Uses ```pickle```'s unpicking features to deserialize and load pickled Python object files (like models, tensor or dictionaries) into memory. You can also set which device to load to object to (CPU or GPU etc.)|
|```torch.nn.Module.load_state_dict```|Loads a model's parameter dictionary (```model.state_dict()```) using a saved ```state_dict()``` object.|

#### How to save a pytorch model?
There are several stpe to save a pytorch model:
1. We'll create a directory for saving models to called ```models``` using python's ```pathlib``` module.
2. We'll create a file path to save the model to.
3. We'll call ```torch.save(obj, filename)``` where ```obj``` is the target model's ```state_dict()``` of where to save the model.
**Recommended way** for saving a model for inference is:
```python3
torch.save(model.state_dict(), PATH)
```
> **Note** : It's common convention for PyTorch saved models or objects to end with ```.pt``` or ```.pth```, like ```saved_model_01.pth```.


These information captured from this [blog](https://www.learnpytorch.io)

কি কেমন দিলাম? :wink: