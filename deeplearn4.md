```python
import torch

x = torch.arange(4.0)
x
```




    tensor([0., 1., 2., 3.])




```python
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
```


```python
y = 2 * torch.dot(x, x)
y
```




    tensor(28., grad_fn=<MulBackward0>)




```python
y.backward()
x.grad
```




    tensor([ 0.,  4.,  8., 12.])




```python
x.grad == 4 * x
```




    tensor([True, True, True, True])




```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```




    tensor([1., 1., 1., 1.])




```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```




    tensor([0., 2., 4., 6.])




```python
x.grad.zero_()          # 清空梯度 
y = x * x               # y = x²（保留梯度路径）
u = y.detach()          # 分离y，得到u（无梯度路径）
z = u * x               # z = u * x（梯度仅依赖x）
z.sum().backward()      # 反向传播，计算dz/dx = u 
print(x.grad == u)      # 验证梯度是否等于u的值（结果为True）
```

    tensor([True, True, True, True])
    


```python
# 这玩意不会高等数学基本上可以出门右拐了
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```




    tensor([True, True, True, True])


