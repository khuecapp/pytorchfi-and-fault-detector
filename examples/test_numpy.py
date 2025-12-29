import numpy as np

def conv2d_int8_acc32(x_i8, w_i8, stride=1, padding=1):
    """
    x_i8: (N, Cin, H, W) dtype int8
    w_i8: (Cout, Cin, kH, kW) dtype int8
    return: acc (N, Cout, Hout, Wout) dtype int32
    """
    assert x_i8.dtype == np.int8 and w_i8.dtype == np.int8
    N, Cin, H, W = x_i8.shape
    Cout, Cin2, kH, kW = w_i8.shape
    assert Cin == Cin2

    # pad input (pad value = 0 đúng với signed int8)
    x_pad = np.pad(x_i8, ((0,0),(0,0),(padding,padding),(padding,padding)), mode="constant", constant_values=0)

    Hpad, Wpad = x_pad.shape[2], x_pad.shape[3]
    Hout = (Hpad - kH)//stride + 1
    Wout = (Wpad - kW)//stride + 1

    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)

    # loops (rõ ràng, dễ kiểm soát)
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                for wo in range(Wout):
                    h0 = ho * stride
                    w0 = wo * stride
                    patch = x_pad[n, :, h0:h0+kH, w0:w0+kW].astype(np.int32)
                    kernel = w_i8[co].astype(np.int32)
                    out[n, co, ho, wo] = np.sum(patch * kernel, dtype=np.int32)

    return out

def flip_bit_int32(val_i32: np.int32, bit: int) -> np.int32:
    """Lật 1 bit (0..31) của một giá trị int32."""
    assert 0 <= bit < 32
    u = np.uint32(val_i32.view(np.uint32))      # nhìn như unsigned để thao tác bit
    u ^= (np.uint32(1) << np.uint32(bit))       # XOR để lật bit
    return u.view(np.int32)                     # quay lại signed int32

def inject_bitflip_output_neuron(y_acc: np.ndarray,
                                 idx=None,
                                 bit=None,
                                 rng=None):

    assert y_acc.dtype == np.int32
    assert y_acc.ndim == 4

    if rng is None:
        rng = np.random.default_rng()

    y_corrupt = y_acc.copy()

    # Pick neuron
    if idx is None:
        n = int(rng.integers(0, y_acc.shape[0]))
        co = int(rng.integers(0, y_acc.shape[1]))
        ho = int(rng.integers(0, y_acc.shape[2]))
        wo = int(rng.integers(0, y_acc.shape[3]))
        idx = (n, co, ho, wo)
    else:
        n, co, ho, wo = idx

    # Pick bit
    if bit is None:
        bit = int(rng.integers(0, 32))

    before = np.int32(y_corrupt[n, co, ho, wo])
    after = flip_bit_int32(before, bit)
    y_corrupt[n, co, ho, wo] = after

    info = {
        "idx": (n, co, ho, wo),
        "bit": bit,
        "before": int(before),
        "after": int(after),
    }
    return y_corrupt, info


if __name__ == "__main__":
    np.random.seed(0)

    # Ví dụ nhỏ cho dễ nhìn:
    N, Cin, H, W = 1, 3, 13, 13
    Cout, kH, kW = 3, 3, 3
    stride, padding = 1, 1

    # Tạo input/weight int8 (giá trị nhỏ để dễ kiểm tra)
    x = np.random.randint(-5, 6, size=(N, Cin, H, W), dtype=np.int8)
    w = np.random.randint(-3, 4, size=(Cout, Cin, kH, kW), dtype=np.int8)

    y_acc = conv2d_int8_acc32(x, w, stride=stride, padding=padding)
    

    # Lật bit ngẫu nhiên ở 1 neuron ngẫu nhiên
    y_faulty, info = inject_bitflip_output_neuron(y_acc)
    print("Injected:", info)