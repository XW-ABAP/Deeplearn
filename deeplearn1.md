```python
import torch
# 导入 PyTorch 库，PyTorch 是一个用于深度学习和张量计算的强大库

x = torch.arange(12)
# 创建一个包含从 0 到 11 的整数的一维张量 x
# torch.arange 函数用于生成一个指定范围的整数序列

print(x)
# 打印张量 x 的内容

print(x.shape)
# 打印张量 x 的形状
# 形状表示张量在每个维度上的大小，这里输出 (12,) 表示一维张量有 12 个元素

x.numel()
# 获取张量 x 中元素的总数
# 这里 x 有 12 个元素，但代码未对该结果进行打印输出

X = x.reshape(3, 4)
# 将一维张量 x 重塑为一个 3 行 4 列的二维张量 X
# reshape 方法在不改变元素总数的情况下改变张量的形状

print(X)
# 打印重塑后的二维张量 X

Xzero = torch.zeros((2, 3, 4))
# 创建一个形状为 (2, 3, 4) 的三维张量 Xzero
# torch.zeros 函数用于创建所有元素都为 0 的张量

print(Xzero)
# 打印全零的三维张量 Xzero

Xone = torch.ones((2, 3, 4))
# 创建一个形状为 (2, 3, 4) 的三维张量 Xone
# torch.ones 函数用于创建所有元素都为 1 的张量

print(Xone)
# 打印全一的三维张量 Xone


# 有时我们想通过从某个特定的概率分布中随机采样来得到张量中每个元素的值。例如，当我们构造数组来作
# 为神经网络中的参数时，我们通常会随机初始化参数的值。以下代码创建一个形状为（3,4）的张量。其中的
# 每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
Xrandn = torch.randn(3, 4)
# 创建一个形状为 (3, 4) 的二维张量 Xrandn
# torch.randn 函数用于创建元素从标准正态分布（均值为 0，标准差为 1）中随机采样得到的张量

print(Xrandn)
# 打印随机生成的二维张量 Xrandn




xtensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(xtensor)
```

    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    torch.Size([12])
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    tensor([[[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
    
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]])
    tensor([[[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],
    
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    tensor([[ 0.9768,  0.2387, -0.6481, -0.8569],
            [ 1.5381,  0.2890,  0.2874, -1.7051],
            [ 1.8424,  1.9824,  1.2392,  1.1268]])
    tensor([[2, 1, 4, 3],
            [1, 2, 3, 4],
            [4, 3, 2, 1]])
    


```python
import torch
# 导入 PyTorch 库，PyTorch 是一个用于深度学习和张量计算的强大库


x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])

print(f"{x + y}\n{x - y}\n{x * y}\n{x / y}\n{x ** y}")

print(torch.exp(x))
```

    tensor([ 3.,  4.,  6., 10.])
    tensor([-1.,  0.,  2.,  6.])
    tensor([ 2.,  4.,  8., 16.])
    tensor([0.5000, 1.0000, 2.0000, 4.0000])
    tensor([ 1.,  4., 16., 64.])
    tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
    


```python
import torch
# 导入 PyTorch 库，用于张量计算和深度学习

# 创建第一个张量 X
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# 解释：
# 1. torch.arange(12) 生成一个从 0 到 11 的整数序列
# 2. dtype=torch.float32 将数据类型转换为 32 位浮点数
# 3. reshape((3,4)) 将一维张量重塑为 3 行 4 列的二维张量
# 输出示例：
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])

# 创建第二个张量 Y
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 解释：
# 手动创建一个 3 行 4 列的二维张量，元素为浮点数
# 输出示例：
# tensor([[2., 1., 4., 3.],
#         [1., 2., 3., 4.],
#         [4., 3., 2., 1.]])

# 沿不同维度拼接张量 X 和 Y
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# 解释：
# 1. torch.cat 是张量拼接函数
# 2. (X, Y) 表示要拼接的两个张量
# 3. dim 参数控制拼接维度：
#    - dim=0（垂直拼接）：在行方向拼接，要求列数相同
#    - dim=1（水平拼接）：在列方向拼接，要求行数相同
# 输出示例：
# 垂直拼接 (dim=0):
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])

# 水平拼接 (dim=1):
# tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])


X == Y



```




    tensor([[False,  True, False,  True],
            [False, False, False, False],
            [False, False, False, False]])




```python
X.sum()
```




    tensor(66.)




```python
a = torch.arange(3).reshape((3, 1))
# 创建一个包含从 0 到 2 的整数的一维张量，然后将其重塑为形状为 (3, 1) 的二维张量
# 也就是说，会得到一个 3 行 1 列的二维张量，元素依次是 0、1、2
# 示例输出：
# tensor([[0],
#         [1],
#         [2]])

b = torch.arange(2).reshape((1, 2))
# 创建一个包含从 0 到 1 的整数的一维张量，接着将其重塑为形状为 (1, 2) 的二维张量
# 也就是得到一个 1 行 2 列的二维张量，元素依次是 0、1
# 示例输出：
# tensor([[0, 1]])

a, b
# 这里会同时返回张量 a 和张量 b 的内容
# 可以通过打印语句来查看结果，例如 print(a, b) 


a + b
```




    tensor([[0, 1],
            [1, 2],
            [2, 3]])




```python
X[-1], X[1:3]
```




    (tensor([ 8.,  9., 10., 11.]),
     tensor([[ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]]))




```python
X[1, 2] = 9
X
```




    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  9.,  7.],
            [ 8.,  9., 10., 11.]])




```python
X[0:2, :] = 12
X
```




    tensor([[12., 12., 12., 12.],
            [12., 12., 12., 12.],
            [ 8.,  9., 10., 11.]])




```python
before = id(Y)
Y = Y + X
id(Y) == before
```




    False




```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

    id(Z): 1988851544016
    id(Z): 1988851544016
    


```python
before = id(X)
X += Y
id(X) == before
```




    True




```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```




    (numpy.ndarray, torch.Tensor)


