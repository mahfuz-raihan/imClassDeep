# imClassDeep
This will use for pytorch note to determine the functionality of the pytorch and inner function

#### 3 step to making predictions with pytorch model:
1. Set the model with evaluation mode (```model.eval()```)
2. Make the predictions using inference mode context manager (```with torch.inference_mode():....```)
3. All prediction should be made with objects on the same device (weather data and model on GPU or CPU only)

### How to save and load a pytorch model? 
First we need to save a pytorch model, after that we can load the model for further work. Let's see how to save a pytorch model
The method saving and loading model in pytorch:
|pyTroch method|What does it do?|
|--------------|----------------|
#### How to save a pytorch model?
There are several stpe to save a pytorch model