"""
An easy way to find unused params is train your model 
on a single node without the DDP wrapper. after loss.backward() 
and before optimizer.step() call add the below lines

This will print any param which did not get used in loss calculation, their grad will be None.
https://discuss.pytorch.org/t/how-to-find-the-unused-parameters-in-network/63948/4
"""

for name, param in model.named_parameters():
    if param.grad is None:
        print(name)
