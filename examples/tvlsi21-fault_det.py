import torch
import torch.nn.functional as F

# torch.manual_seed(5)
IN_SIZE = 13
KER_SIZE = 3
IN_CH = 3
OUT_CH = 3

# Input
x = torch.rand(1, IN_CH, IN_SIZE, IN_SIZE)
x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0) # padded input
w = torch.rand(OUT_CH, IN_CH, KER_SIZE, KER_SIZE) # kernel

C = x_padded.shape[1]
H = x_padded.shape[2]    
W = x_padded.shape[3]
M = KER_SIZE
# Input checksum computation
in_channel_sum = torch.zeros(C, 1, 1)

for ch in range(IN_CH):
    ch_sum = x_padded[0, ch, :, :]
    in_channel_sum[ch] = ch_sum.sum()

window_sum = torch.zeros(C, KER_SIZE, KER_SIZE)
window_sub = torch.zeros(C, KER_SIZE, KER_SIZE)
window_sum2 = torch.zeros(C, KER_SIZE, KER_SIZE)

for ch in range(IN_CH):
    for i in range(W):
        for j in range(H):
            for m in range(KER_SIZE):
                for n in range(KER_SIZE):
                    if m>i or i>H-M+m or n>j or j>W-M+n:
                        window_sub[ch, m, n] += x_padded[0, ch, i, j]
                        
for ch in range(IN_CH):
    for i in range(KER_SIZE):
        for j in range(KER_SIZE):
            window_sum[ch, i, j] = in_channel_sum[ch] - window_sub[ch, i, j]

for ch in range(IN_CH):
    for i in range(KER_SIZE):
        for j in range(KER_SIZE):
            window =  x_padded[0, ch, i:i+H-2, j:j+W-2]
            window_sum2[ch, i, j] = window.sum()

mul_lst = torch.zeros(OUT_CH, IN_CH, KER_SIZE, KER_SIZE)
mul_lst2 = torch.zeros(OUT_CH, IN_CH, KER_SIZE, KER_SIZE)
for out_c in range(OUT_CH):
    for in_c in range(IN_CH):
        for i in range(KER_SIZE):
            for j in range(KER_SIZE):
                mul_lst[out_c, in_c, i, j] = w[out_c, in_c, i, j] * window_sum[in_c, i, j]
                mul_lst2[out_c, in_c, i, j] = w[out_c, in_c, i, j] * window_sum2[in_c, i, j]

input_checksum = torch.zeros(OUT_CH, 1)
input_checksum2 = torch.zeros(OUT_CH, 1)
for out_c in range(OUT_CH):
    input_checksum[out_c] = mul_lst[out_c, :, :, :].sum()
    input_checksum2[out_c] = mul_lst2[out_c, :, :, :].sum()

# Convolution operation
def custom_conv2d(input_tensor, weight_tensor, bias_tensor=None, stride=1, padding=0):
    output_tensor = F.conv2d(input_tensor, weight_tensor, bias=bias_tensor, stride=stride, padding=padding)    
    return output_tensor
y = custom_conv2d(x_padded, w)

output_checksum = torch.zeros(OUT_CH, 1)
for out_c in range(OUT_CH):
    output_checksum[out_c] = y[0, out_c, :, :].sum()

print(f"input_checksum = {input_checksum}")
print(f"output_checksum = {output_checksum}")

# Fault detection
threshold = 1e-6 
errors = torch.zeros(OUT_CH, 1, dtype=torch.bool)

for out_c in range(OUT_CH):
    abs_diff = torch.abs(input_checksum[out_c] - output_checksum[out_c])
    
    # Relative error: diff / magnitude
    magnitude = torch.abs(input_checksum[out_c]) + 1e-10  # Avoid division by zero
    relative_error = abs_diff / magnitude
    
    # Convert to scalar for printing
    print(f"Channel {out_c}: abs_diff={abs_diff.item():.2e}, rel_error={relative_error.item():.2e}")
    
    if relative_error > 1e-4:  # 0.01% error threshold
        errors[out_c] = True
print(f"errors = {errors}")
    
            