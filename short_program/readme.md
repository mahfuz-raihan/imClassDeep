### How to check the content of a pytorch model

```python
# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel() # you can make your own and here's just a example model instance

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())
```

These information captured from this [blog](https://www.learnpytorch.io)