import torch
import torch.nn.functional as F

torch.manual_seed(0)
IN_SIZE = 5
KER_SIZE = 3

# Input
x = torch.randn(1, 1, IN_SIZE, IN_SIZE)

H = x.shape[2]    
W = x.shape[3]
print(f"x = {x}")
#result = []
result_ = torch.zeros(1, 1, KER_SIZE, KER_SIZE)
for i in range(KER_SIZE):
    for j in range(KER_SIZE):
        window =  x[0, 0, i:i+H-2, j:j+W-2]
        print(window)      
        window_sum = window.sum()
        #result.append(window_sum)
        result_[0, 0, i, j] = window_sum
#print(f"result = {result}")
print(f"result_ = {result_}")
# Kernel
w = torch.randn(1, 1, KER_SIZE, KER_SIZE)
print(f"w = {w}")

mul_lst = torch.zeros(1, 1, KER_SIZE, KER_SIZE)
for i in range(KER_SIZE):
    for j in range(KER_SIZE):
        mul_lst[0, 0, i, j] = w[0, 0, i, j] * result_[0, 0, i, j]
y = F.conv2d(x, w, bias=None, stride=1, padding=0)
print(f"mul_lst = {mul_lst}")
input_checksum = mul_lst.sum()
output_checksum = y.sum()
print(f"y = {y}")
print(f"input_checksum = {input_checksum}")
print(f"output_checksum = {output_checksum}")