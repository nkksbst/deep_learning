import matplotlib.pyplot as plt
import numpy as np
import torch

"""
Single plot
"""
x = [1,2,3]
y = [2,4,6]

# OO Style plotting
fig, ax = plt.subplots()  
ax.plot(x, y, label='linear')  
ax.plot(x, [i ** 2 for i in y], label='quadratic')  
ax.plot(x, [i ** 3 for i in y], label='cubic')  
ax.set_xlabel('x')  
ax.set_ylabel('y')  
ax.set_title("Simple Plot")  
ax.legend() 

plt.savefig('sample_single.png')

"""
Multiple plots
"""
n_rows = 2
n_cols = 1
index = 1
plt.subplot(n_rows, n_cols, index)
plt.plot(x, y)
plt.title('Sample Multiple Plots')
plt.ylabel('y')
index = 2
plt.subplot(2, 1, 2)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('sample_multi.png')

"""
Scatter Plot
"""
plt.subplots() 
plt.scatter(x, y, color = '#88c999')
plt.savefig('sample_scatter.png')

"""
plotting a tensor
"""
im = torch.randn(3,24,24)
plt.imshow(np.transpose(im.numpy(), (1,2,0)))