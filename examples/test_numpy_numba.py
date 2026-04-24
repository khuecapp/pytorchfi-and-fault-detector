import numpy as np
import time
from numba import jit

# -------- Bit-flip helpers (Numba-compatible) --------
@jit(nopython=True)
def flip_bit_int8_numba(val_i8, bit):
    """Flip 1 bit (0..7) of int8 value."""
    u = np.uint8(val_i8)
    u ^= np.uint8(1) << bit
    return np.int8(u)

@jit(nopython=True)
def flip_bit_int64_numba(val_i64, bit):
    """Flip 1 bit (0..63) of int64 value."""
    u = np.uint64(val_i64)
    u ^= np.uint64(1) << bit
    return np.int64(u)

@jit(nopython=True)
def flip_bit_int32_numba(val_i32, bit):
    """Flip 1 bit (0..31) of int32 value."""
    u = np.uint32(val_i32)
    u ^= np.uint32(1) << bit
    return np.int32(u)

# Python wrappers for non-numba code
def flip_bit_int8(val_i8: np.int8, bit: int) -> np.int8:
    """Flip 1 bit (0..7) of int8 value."""
    assert 0 <= bit < 8
    u = np.uint8(val_i8.view(np.uint8))
    u ^= (np.uint8(1) << np.uint8(bit))
    return u.view(np.int8)

def flip_bit_int32(val_i32: np.int32, bit: int) -> np.int32:
    """Flip 1 bit (0..31) of int32 value."""
    assert 0 <= bit < 32
    u = np.uint32(val_i32.view(np.uint32))
    u ^= (np.uint32(1) << np.uint32(bit))
    return u.view(np.int32)

def flip_bit_int64(val_i64: np.int64, bit: int) -> np.int64:
    """Flip 1 bit (0..63) of int64 value."""
    assert 0 <= bit < 64
    u = np.uint64(val_i64.view(np.uint64))
    u ^= (np.uint64(1) << np.uint64(bit))
    return u.view(np.int64)

# -------- Fault spec utilities --------
def random_seu_fault_spec(out_shape, Cin, kH, kW, rng=None, kind="acc"):
    """
    Create a random SEU fault spec targeting a single output element and a single tap.
    kind: "act" | "w" | "acc"
    """
    if rng is None:
        rng = np.random.default_rng()

    N, Cout, Hout, Wout = out_shape
    out_idx = (
        int(rng.integers(0, N)),
        int(rng.integers(0, Cout)),
        int(rng.integers(0, Hout)),
        int(rng.integers(0, Wout)),
    )
    tap = (
        int(rng.integers(0, Cin)),
        int(rng.integers(0, kH)),
        int(rng.integers(0, kW)),
    )

    if kind in ("act", "w"):
        bit = int(rng.integers(0, 8))      # int8 bit
    elif kind == "acc":
        bit = int(rng.integers(0, 32))     # int32 bit
    else:
        raise ValueError("kind must be 'act', 'w', or 'acc'")

    return {"kind": kind, "out_idx": out_idx, "tap": tap, "bit": bit}


# -------- Numba-accelerated convolution kernels --------
@jit(nopython=True)
def conv2d_fast_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride):
    """Fast path: no fault injection, pure computation"""
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)
    
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                for wo in range(Wout):
                    h0 = ho * stride
                    w0 = wo * stride
                    acc = np.int64(0)
                    
                    for ci in range(Cin):
                        for kh in range(kH):
                            for kw in range(kW):
                                a = np.int64(x_pad[n, ci, h0 + kh, w0 + kw])
                                b = np.int64(w_i8[co, ci, kh, kw])
                                acc += a * b
                    
                    out[n, co, ho, wo] = np.int32(acc)
    
    return out

@jit(nopython=True)
def conv2d_with_fault_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                            fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                            fault_ci, fault_kh, fault_kw, fault_bit):
    """
    Slow path: with fault injection
    
    Weight-reuse architecture:
    - Weight fault: affects ALL spatial locations in the tile (entire Hout x Wout)
    - Activation fault: affects only 1 specific MAC operation
    - Accumulator fault: affects only 1 output element
    """
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)
    
    injected = False
    inject_before = np.int64(0)
    inject_after = np.int64(0)
    
    # Pre-load and potentially corrupt weight (weight-reuse)
    w_corrupted = np.zeros_like(w_i8)
    for co in range(Cout):
        for ci in range(Cin):
            for kh in range(kH):
                for kw in range(kW):
                    w_val = w_i8[co, ci, kh, kw]

                    # Weight fault: inject once, affects all spatial locations
                    if (not injected) and (fault_kind == 1):  # 1 = "w"
                        if (co == fault_co and ci == fault_ci and 
                            kh == fault_kh and kw == fault_kw):
                            print(f"fault injection at weight[{co},{ci},{kh},{kw}]")
                            inject_before = np.int64(w_val)
                            w_val = flip_bit_int8_numba(w_val, fault_bit)
                            inject_after = np.int64(w_val)
                            print(f"injecting weight fault: before={inject_before}, after={inject_after}")
                            injected = True
                    
                    w_corrupted[co, ci, kh, kw] = w_val
    
    # Convolution with potentially corrupted weights
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                for wo in range(Wout):
                    h0 = ho * stride
                    w0 = wo * stride
                    acc = np.int64(0)
                    
                    for ci in range(Cin):
                        for kh in range(kH):
                            for kw in range(kW):
                                # Load activation (fresh for each MAC)
                                a_i8 = np.int8(x_pad[n, ci, h0 + kh, w0 + kw])
                                
                                # Inject activation fault (affects only this specific MAC)
                                if (not injected) and (fault_kind == 0):  # 0 = "act"
                                    if (n == fault_n and co == fault_co and 
                                        ho == fault_ho and wo == fault_wo and
                                        ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = np.int64(a_i8)
                                        a_i8 = flip_bit_int8_numba(a_i8, fault_bit)
                                        inject_after = np.int64(a_i8)
                                        injected = True
                                
                                # Load weight (from potentially corrupted array)
                                b_i8 = w_corrupted[co, ci, kh, kw]
                                
                                # Compute product
                                prod = np.int64(a_i8) * np.int64(b_i8)
                                acc = acc + prod
                                
                                # Inject accumulator fault (affects only this output element)
                                if (not injected) and (fault_kind == 2):  # 2 = "acc"
                                    if (n == fault_n and co == fault_co and 
                                        ho == fault_ho and wo == fault_wo and
                                        ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = acc
                                        acc = flip_bit_int64_numba(acc, fault_bit)
                                        inject_after = acc
                                        injected = True
                    
                    out[n, co, ho, wo] = np.int32(acc)
    
    return out, inject_before, inject_after


# -------- Frame-based reuse kernels (loop: n→co→ci→ho→wo→kh→kw) --------
@jit(nopython=True)
def conv2d_frame_reuse_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride):
    """Frame-based reuse: weight block w[co,ci,:,:] reused for all (ho,wo) before moving to next ci."""
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)
    for n in range(N):
        for co in range(Cout):
            for ci in range(Cin):
                for ho in range(Hout):
                    h0 = ho * stride
                    for wo in range(Wout):
                        w0 = wo * stride
                        for kh in range(kH):
                            for kw in range(kW):
                                a = np.int32(x_pad[n, ci, h0 + kh, w0 + kw])
                                b = np.int32(w_i8[co, ci, kh, kw])
                                out[n, co, ho, wo] += a * b
    return out

@jit(nopython=True)
def conv2d_frame_reuse_with_fault_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                                        fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                                        fault_ci, fault_kh, fault_kw, fault_bit):
    """Frame-based reuse with fault injection (loop: n→co→ci→ho→wo→kh→kw)."""
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)

    injected = False
    inject_before = np.int64(0)
    inject_after = np.int64(0)

    # Pre-load and potentially corrupt weights
    w_corrupted = np.zeros_like(w_i8)
    for co in range(Cout):
        for ci in range(Cin):
            for kh in range(kH):
                for kw in range(kW):
                    w_val = w_i8[co, ci, kh, kw]
                    if (not injected) and (fault_kind == 1):
                        if (co == fault_co and ci == fault_ci and
                                kh == fault_kh and kw == fault_kw):
                            inject_before = np.int64(w_val)
                            w_val = flip_bit_int8_numba(w_val, fault_bit)
                            inject_after = np.int64(w_val)
                            injected = True
                    w_corrupted[co, ci, kh, kw] = w_val

    for n in range(N):
        for co in range(Cout):
            for ci in range(Cin):
                for ho in range(Hout):
                    h0 = ho * stride
                    for wo in range(Wout):
                        w0 = wo * stride
                        for kh in range(kH):
                            for kw in range(kW):
                                a_i8 = np.int8(x_pad[n, ci, h0 + kh, w0 + kw])

                                if (not injected) and (fault_kind == 0):
                                    if (n == fault_n and co == fault_co and
                                            ho == fault_ho and wo == fault_wo and
                                            ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = np.int64(a_i8)
                                        a_i8 = flip_bit_int8_numba(a_i8, fault_bit)
                                        inject_after = np.int64(a_i8)
                                        injected = True

                                b_i8 = w_corrupted[co, ci, kh, kw]
                                out[n, co, ho, wo] += np.int32(a_i8) * np.int32(b_i8)

                                if (not injected) and (fault_kind == 2):
                                    if (n == fault_n and co == fault_co and
                                            ho == fault_ho and wo == fault_wo and
                                            ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = np.int64(out[n, co, ho, wo])
                                        out[n, co, ho, wo] = flip_bit_int32_numba(out[n, co, ho, wo], fault_bit)
                                        inject_after = np.int64(out[n, co, ho, wo])
                                        injected = True

    return out, inject_before, inject_after


# -------- Row-based reuse kernels (loop: n→co→ho→ci→wo→kh→kw) --------
@jit(nopython=True)
def conv2d_row_reuse_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride):
    """Row-based reuse: weights preloaded per row, reused across wo; then move to next ci in same row."""
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                h0 = ho * stride
                for ci in range(Cin):
                    for wo in range(Wout):
                        w0 = wo * stride
                        for kh in range(kH):
                            for kw in range(kW):
                                a = np.int32(x_pad[n, ci, h0 + kh, w0 + kw])
                                b = np.int32(w_i8[co, ci, kh, kw])
                                out[n, co, ho, wo] += a * b
    return out

@jit(nopython=True)
def conv2d_row_reuse_with_fault_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                                      fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                                      fault_ci, fault_kh, fault_kw, fault_bit):
    """Row-based reuse with fault injection (loop: n→co→ho→ci→wo→kh→kw)."""
    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)

    injected = False
    inject_before = np.int64(0)
    inject_after = np.int64(0)

    # Pre-load and potentially corrupt weights
    w_corrupted = np.zeros_like(w_i8)
    for co in range(Cout):
        for ci in range(Cin):
            for kh in range(kH):
                for kw in range(kW):
                    w_val = w_i8[co, ci, kh, kw]
                    if (not injected) and (fault_kind == 1):
                        if (co == fault_co and ci == fault_ci and
                                kh == fault_kh and kw == fault_kw):
                            inject_before = np.int64(w_val)
                            w_val = flip_bit_int8_numba(w_val, fault_bit)
                            inject_after = np.int64(w_val)
                            injected = True
                    w_corrupted[co, ci, kh, kw] = w_val

    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                h0 = ho * stride
                for ci in range(Cin):
                    for wo in range(Wout):
                        w0 = wo * stride
                        for kh in range(kH):
                            for kw in range(kW):
                                a_i8 = np.int8(x_pad[n, ci, h0 + kh, w0 + kw])

                                if (not injected) and (fault_kind == 0):
                                    if (n == fault_n and co == fault_co and
                                            ho == fault_ho and wo == fault_wo and
                                            ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = np.int64(a_i8)
                                        a_i8 = flip_bit_int8_numba(a_i8, fault_bit)
                                        inject_after = np.int64(a_i8)
                                        injected = True

                                b_i8 = w_corrupted[co, ci, kh, kw]
                                out[n, co, ho, wo] += np.int32(a_i8) * np.int32(b_i8)

                                if (not injected) and (fault_kind == 2):
                                    if (n == fault_n and co == fault_co and
                                            ho == fault_ho and wo == fault_wo and
                                            ci == fault_ci and kh == fault_kh and kw == fault_kw):
                                        inject_before = np.int64(out[n, co, ho, wo])
                                        out[n, co, ho, wo] = flip_bit_int32_numba(out[n, co, ho, wo], fault_bit)
                                        inject_after = np.int64(out[n, co, ho, wo])
                                        injected = True

    return out, inject_before, inject_after


# -------- Main convolution function --------
def conv2d_int8_acc32_seu(x_i8, w_i8, stride=1, padding=1, fault=None, reuse_scheme="frame"):
    """
    x_i8: (N, Cin, H, W) int8
    w_i8: (Cout, Cin, kH, kW) int8
    out:  (N, Cout, Hout, Wout) int32 accumulator

    fault: None (no fault) OR dict with:
      {
        "kind": "act" | "w" | "acc",
        "out_idx": (n, co, ho, wo),
        "tap": (ci, kh, kw),
        "bit": int
      }
    reuse_scheme: "legacy" | "frame" | "row"
    """
    assert x_i8.dtype == np.int8 and w_i8.dtype == np.int8
    N, Cin, H, W = x_i8.shape
    Cout, Cin2, kH, kW = w_i8.shape
    assert Cin == Cin2

    # Pad input
    x_pad = np.pad(
        x_i8,
        ((0,0),(0,0),(padding,padding),(padding,padding)),
        mode="constant",
        constant_values=0
    )

    Hpad, Wpad = x_pad.shape[2], x_pad.shape[3]
    Hout = (Hpad - kH)//stride + 1
    Wout = (Wpad - kW)//stride + 1

    # Fast path: no fault
    if fault is None:
        if reuse_scheme == "frame":
            out = conv2d_frame_reuse_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride)
        elif reuse_scheme == "row":
            out = conv2d_row_reuse_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride)
        else:  # "legacy"
            out = conv2d_fast_numba(x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride)
        return out, None

    # Slow path: with fault injection
    else:
        # Convert fault kind to int for numba
        fault_kind_map = {"act": 0, "w": 1, "acc": 2}
        fault_kind = fault_kind_map[fault["kind"]]

        fault_n, fault_co, fault_ho, fault_wo = fault["out_idx"]
        fault_ci, fault_kh, fault_kw = fault["tap"]
        fault_bit = fault["bit"]

        if reuse_scheme == "frame":
            out, inject_before, inject_after = conv2d_frame_reuse_with_fault_numba(
                x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                fault_ci, fault_kh, fault_kw, fault_bit
            )
        elif reuse_scheme == "row":
            out, inject_before, inject_after = conv2d_row_reuse_with_fault_numba(
                x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                fault_ci, fault_kh, fault_kw, fault_bit
            )
        else:  # "legacy"
            out, inject_before, inject_after = conv2d_with_fault_numba(
                x_pad, w_i8, N, Cin, Cout, Hout, Wout, kH, kW, stride,
                fault_kind, fault_n, fault_co, fault_ho, fault_wo,
                fault_ci, fault_kh, fault_kw, fault_bit
            )
        
        # Reconstruct inject_info
        inject_info = {
            "kind": fault["kind"],
            "out_idx": fault["out_idx"],
            "tap": fault["tap"],
            "bit": fault["bit"],
            "before": int(inject_before),
            "after": int(inject_after),
        }
        
        return out, inject_info


# -------- Checksum detector --------
def checksum_detector_conv_int8_acc32(
    x_i8: np.ndarray,
    w_i8: np.ndarray,
    y_acc: np.ndarray,
    stride: int = 1,
    padding: int = 1,
    bias_i32: np.ndarray | None = None,
    tol: int = 0,
    method: str = "ours",
):
    """Checksum-based fault detector for conv (int8 x int8 -> int32 acc)."""
    assert x_i8.dtype == np.int8 and w_i8.dtype == np.int8
    assert y_acc.dtype == np.int32
    assert x_i8.ndim == 4 and w_i8.ndim == 4 and y_acc.ndim == 4
    assert stride == 1, "Checksum này đang giả sử stride=1."

    N, Cin, H, W = x_i8.shape
    Cout, Cin2, kH, kW = w_i8.shape
    assert Cin == Cin2

    # pad input
    x_pad = np.pad(
        x_i8,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    _, _, Hout, Wout = y_acc.shape

    # sanity check geometry
    Hpad, Wpad = x_pad.shape[2], x_pad.shape[3]
    exp_Hout = (Hpad - kH) // stride + 1
    exp_Wout = (Wpad - kW) // stride + 1
    assert (Hout, Wout) == (exp_Hout, exp_Wout), "y_acc shape mismatch."

    input_checksum = np.zeros((N, Cout), dtype=np.int64)
    output_checksum = np.zeros((N, Cout), dtype=np.int64)

    w64 = w_i8.astype(np.int64)

    for n in range(N):
        # sum_shift[ci,kh,kw]
        sum_shift = np.zeros((Cin, kH, kW), dtype=np.int64)

        if method.lower() == "ours":
            # OURS
            for ci in range(Cin):
                for kh in range(kH):
                    for kw in range(kW):
                        window = x_pad[n, ci, kh:kh + Hout, kw:kw + Wout].astype(np.int64)
                        sum_shift[ci, kh, kw] = window.sum()

        # input_checksum[co] = sum_{ci,kh,kw} w[co,ci,kh,kw] * sum_shift[ci,kh,kw]
        for co in range(Cout):
            input_checksum[n, co] = np.sum(w64[co] * sum_shift)

    # --- Output checksum ---
    for n in range(N):
        for co in range(Cout):
            out_sum = y_acc[n, co].astype(np.int64).sum()
            # if bias_i32 is not None:
            #     out_sum -= np.int64(bias_i32[co]) * np.int64(Hout) * np.int64(Wout)
            output_checksum[n, co] = out_sum

    diff = input_checksum - output_checksum
    detected = np.abs(diff) > np.int64(tol)
    
    print(f"Diff: {diff}")
    if True in detected:
        res = True  # Fault detected
    else:
        res = False # No fault detected
    
    
    locations = np.where(detected[0])[0]

    info = {
        "method": method,
        "detected": res,
        "fault locations": locations
    }
    if res == False:
        print(f"sum_shift: {sum_shift}")
        print(f"info: {info}")
    
    return res, info

def checksum_detector_conv_int8_acc32_with_fault(
    x_i8: np.ndarray,
    w_i8: np.ndarray,
    y_acc: np.ndarray,
    stride: int = 1,
    padding: int = 1,
    bias_i32: np.ndarray | None = None,
    tol: int = 0,
    method: str = "ours",
):
    """
    Checksum-based fault detector với fault injection trong detection module.
    Inject fault ngẫu nhiên vào input_checksum hoặc output_checksum.
    """
    assert x_i8.dtype == np.int8 and w_i8.dtype == np.int8
    assert y_acc.dtype == np.int32
    assert x_i8.ndim == 4 and w_i8.ndim == 4 and y_acc.ndim == 4
    assert stride == 1

    N, Cin, H, W = x_i8.shape
    Cout, Cin2, kH, kW = w_i8.shape
    assert Cin == Cin2

    x_pad = np.pad(x_i8, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant", constant_values=0)
    _, _, Hout, Wout = y_acc.shape

    input_checksum = np.zeros((N, Cout), dtype=np.int64)
    output_checksum = np.zeros((N, Cout), dtype=np.int64)
    w64 = w_i8.astype(np.int64)

    for n in range(N):
        sum_shift = np.zeros((Cin, kH, kW), dtype=np.int64)
        if method.lower() == "ours":
            for ci in range(Cin):
                for kh in range(kH):
                    for kw in range(kW):
                        window = x_pad[n, ci, kh:kh + Hout, kw:kw + Wout].astype(np.int64)
                        sum_shift[ci, kh, kw] = window.sum()
        for co in range(Cout):
            input_checksum[n, co] = np.sum(w64[co] * sum_shift)

    for n in range(N):
        for co in range(Cout):
            out_sum = y_acc[n, co].astype(np.int64).sum()
            if bias_i32 is not None:
                out_sum -= np.int64(bias_i32[co]) * np.int64(Hout) * np.int64(Wout)
            output_checksum[n, co] = out_sum

    # Inject fault ngẫu nhiên vào detection: flip bit trong input_checksum hoặc output_checksum
    rng = np.random.default_rng()
    if rng.random() < 0.5:  # 50% chance flip input hoặc output
        # Flip bit trong input_checksum
        co = rng.integers(0, Cout)
        bit = rng.integers(0, 64)  # int64 có 64 bit
        input_checksum[0, co] = flip_bit_int64(input_checksum[0, co], bit)
        print(f"Fault injected in input_checksum[0,{co}], bit {bit}")
    else:
        # Flip bit trong output_checksum
        co = rng.integers(0, Cout)
        bit = rng.integers(0, 64)
        output_checksum[0, co] = flip_bit_int64(output_checksum[0, co], bit)
        print(f"Fault injected in output_checksum[0,{co}], bit {bit}")

    diff = input_checksum - output_checksum
    detected = np.abs(diff) > np.int64(tol)
    res = True in detected
    locations = np.where(detected[0])[0]
    info = {"method": method, "detected": res, "fault locations": locations}
    return res, info


# ---------------- Fault Probability Calculation ----------------
def calculate_fault_probability_in_detection(area_ratio_conv_to_det):
    """
    Tính xác suất fault injection ở detection module dựa trên tỷ lệ diện tích.
    """
    area_conv = 1.0
    area_det = area_ratio_conv_to_det * area_conv
    total_area = area_conv + area_det
    prob_det = area_det / total_area
    return prob_det


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Tính xác suất fault ở detection module
    K = 4
    area_ratio = (48 + 16*K)/((256 + 256*K) + (48 + 16*K))

    print(area_ratio)
    prob_fault_det = calculate_fault_probability_in_detection(area_ratio)
    print(f"Probability that fault occurs in detection module: {prob_fault_det:.4f} ({prob_fault_det*100:.2f}%)")

    # Config here!
    np.random.seed(0)
    N_RUNS = 10000
    METHOD = "ours"
    KIND = "acc"           # "act", "w", or "acc"
    REUSE_SCHEME = "row"  # "frame", "row", or "legacy"
    
    if KIND == "acc":
        component = "partial sum (accumulator)"
    elif KIND == "w":
        component = "weight"
    elif KIND == "act":
        component = "input activation"
    
    detected, fp, no_fault, conv_fault_runs, det_fault_runs, compensate = 0, 0, 0, 0, 0, 0
    
    print(f"\n=== Starting {N_RUNS} runs with Numba acceleration ===")
    print(f"Fault type: {KIND} ({component})")
    print(f"Detection method: {METHOD}")
    print(f"Reuse scheme: {REUSE_SCHEME}\n")
    
    start = time.perf_counter()
    
    for run in range(N_RUNS):
        N, Cin, H, W = 1, 16, 52, 52
        Cout, kH, kW = 16, 3, 3
        stride, padding = 1, 1
        out_shape = (1, Cout, H, W)

        x = np.random.randint(-5, 6, size=(N, Cin, H, W), dtype=np.int8)
        w = np.random.randint(-3, 4, size=(Cout, Cin, kH, kW), dtype=np.int8)
        w_golden = w.copy()
        # Fault injection decision
        fault_in_detection = np.random.rand() < prob_fault_det
        # Assume faults always occur in conv module for this test
        fault_in_detection = False
        
        if fault_in_detection:
            det_fault_runs += 1
            # Inject fault vào detection module
            print(f"RUN {run + 1}: Fault in Detection Module")
            y, _ = conv2d_int8_acc32_seu(x, w, stride=stride, padding=padding, fault=None, reuse_scheme=REUSE_SCHEME)
            det, info = checksum_detector_conv_int8_acc32_with_fault(x, w, y, stride=1, padding=1, tol=0, method=METHOD)
            if det:
                fp += 1
            else:
                compensate += 1
        else:
            conv_fault_runs += 1
            # Inject fault vào convolution module
            print(f"RUN {run + 1}: Fault in Convolution Module")
            fault = random_seu_fault_spec(out_shape=out_shape, Cin=Cin, kH=kH, kW=kW, kind=KIND)
            y_faulty, info = conv2d_int8_acc32_seu(x, w, stride=stride, padding=padding, fault=fault, reuse_scheme=REUSE_SCHEME)
            det, info = checksum_detector_conv_int8_acc32(x, w_golden, y_faulty, stride=1, padding=1, tol=0, method=METHOD)
            if det:
                detected += 1
                print(det, info)
            else:
                no_fault += 1
                print(det, info)

    end = time.perf_counter()
    elapsed = end - start
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS - {N_RUNS} runs completed in {elapsed:.2f}s")
    print(f"Average time per run: {elapsed/N_RUNS*1000:.2f}ms")
    print(f"{'='*60}")
    print(f"Fault in main computation: {conv_fault_runs}")
    print(f"  - Detected: {detected}")
    print(f"  - Missed: {no_fault}")
    print(f"  - Detection rate: {detected/conv_fault_runs*100:.2f}%")
    print(f"\nDetection module faults: {det_fault_runs}")
    # print(f"  - False positives: {fp}")
    # print(f"  - FP rate: {fp/det_fault_runs*100:.2f}%")
    print(f"\nOverall fault detection rate: {detected/N_RUNS*100:.2f}%")
    print(f"Compensate: {compensate}")
    print(f"{'='*60}")