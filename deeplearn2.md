```python
import torch 
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x ** y

```




    (tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))




```python
x = torch.arange(4)
x
```




    tensor([0, 1, 2, 3])




```python
x[3]

```




    tensor(3)




```python
len(x)
```




    4




```python
x.shape
```




    torch.Size([4])




```python
# 5列4行
A = torch.arange(20).reshape(5,4)
A
```




    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]])




```python
A.T  
# 行列转换
```




    tensor([[ 0,  4,  8, 12, 16],
            [ 1,  5,  9, 13, 17],
            [ 2,  6, 10, 14, 18],
            [ 3,  7, 11, 15, 19]])




```python
B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
B
```




    tensor([[1, 2, 3],
            [2, 0, 4],
            [3, 4, 5]])




```python
B = B.T
B

```




    tensor([[1, 2, 3],
            [2, 0, 4],
            [3, 4, 5]])




```python
X = torch.arange(24).reshape(2,3,4)
X
```




    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])




```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
A, A + B
```




    (tensor([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.],
             [12., 13., 14., 15.],
             [16., 17., 18., 19.]]),
     tensor([[ 0.,  2.,  4.,  6.],
             [ 8., 10., 12., 14.],
             [16., 18., 20., 22.],
             [24., 26., 28., 30.],
             [32., 34., 36., 38.]]))




```python
A * B
```




    tensor([[  0.,   1.,   4.,   9.],
            [ 16.,  25.,  36.,  49.],
            [ 64.,  81., 100., 121.],
            [144., 169., 196., 225.],
            [256., 289., 324., 361.]])




```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```




    (tensor([[[ 2,  3,  4,  5],
              [ 6,  7,  8,  9],
              [10, 11, 12, 13]],
     
             [[14, 15, 16, 17],
              [18, 19, 20, 21],
              [22, 23, 24, 25]]]),
     torch.Size([2, 3, 4]))




```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```




    (tensor([0., 1., 2., 3.]), tensor(6.))




```python
A.shape, A.sum()
```




    (torch.Size([5, 4]), tensor(190.))




```python
A_sum_axis0 = A.sum(axis=0)       #列累加
A_sum_axis0, A_sum_axis0.shape
```




    (tensor([40., 45., 50., 55.]), torch.Size([4]))




```python
A_sum_axis1 = A.sum(axis=1)      #行累加
A_sum_axis1, A_sum_axis1.shape
```




    (tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))




```python
A.sum(axis=[0, 1]) # 结果和A.sum()相同
```




    tensor(190.)




```python
A.mean(), A.sum() / A.numel()
```




    (tensor(9.5000), tensor(9.5000))




```python
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```




    (tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))




```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```




    tensor([[ 6.],
            [22.],
            [38.],
            [54.],
            [70.]])




```python
A / sum_A
```




    tensor([[0.0000, 0.1667, 0.3333, 0.5000],
            [0.1818, 0.2273, 0.2727, 0.3182],
            [0.2105, 0.2368, 0.2632, 0.2895],
            [0.2222, 0.2407, 0.2593, 0.2778],
            [0.2286, 0.2429, 0.2571, 0.2714]])




```python
A.cumsum(axis=0)
```




    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  6.,  8., 10.],
            [12., 15., 18., 21.],
            [24., 28., 32., 36.],
            [40., 45., 50., 55.]])




```python
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```




    (tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))




```python
torch.sum(x * y)
```




    tensor(6.)




```python
A.shape, x.shape, torch.mv(A, x)
```




    (torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))




```python
A

```




    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.]])




```python
B = torch.ones(4, 3)
B
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])



$C_{ij} = \sum_{k = 1}^{n} A_{ik} \cdot B_{kj}$


有点复杂  还好我脑瓜子灵活


```python
B = torch.ones(4, 3)
torch.mm(A, B)  #这个有点复杂 
```




    tensor([[ 6.,  6.,  6.],
            [22., 22., 22.],
            [38., 38., 38.],
            [54., 54., 54.],
            [70., 70., 70.]])


