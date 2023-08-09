# Computer Vision note with pytorch based

So, computer vision is a art of teaching what how a computer see the real world. For example, it involve to build a model that classify a photo weather it is cat or dog which we call [binary classification](https://developers.google.com/machine-learning/glossary#binary-classification)

or two or more which is call [multi class classification](https://developers.google.com/machine-learning/glossary#multi-class-classification)

or mark off any object in the video frame which is called [object detection](https://en.wikipedia.org/wiki/Object_detection)

or figure out the different object in a image can be seperated which is called [panoptic segmentation](https://arxiv.org/abs/1801.00868)


## CV libraries in pytorch
| Pytorch module  | Description  |
|:---------------:|:------------:|
| ```torchvision```   | ```torchvision``` module contains datasets, model architrcture and image transformations often used in computer vision problem.  |
| ```torchvision.datasets```  | we can find build-in computer vision datasets for image classfication, object detection, image captioning, video classfication and more.  |
|  ```torchvision.models```  | ```torchvision.models``` contains well-performing and commonly used cv model architectures implemented in pytorch, we can use these with our own problems.  |
|```torch.utils.data.Dataset```  | base dataset class for pytorch.  |
| ```torch.utils.data.DataLoader```  | Creates a python iterable over a dataset (created with ```torch.utils.data.Dataset```).  |
| ```torchvision.transforms``` | image need to be transformed (that turned into numbers/processed/augmented) before being used with a model, here common image transformation are found.|


Now, let's go to the excercise.....
