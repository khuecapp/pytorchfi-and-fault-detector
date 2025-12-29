import numpy as np

# -------- Bit-flip helpers (safe for signed ints) --------
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


# -------- Convolution with register-level SEU injection --------
def conv2d_int8_acc32_seu(x_i8, w_i8, stride=1, padding=1, fault=None):
    """
    x_i8: (N, Cin, H, W) int8
    w_i8: (Cout, Cin, kH, kW) int8
    out:  (N, Cout, Hout, Wout) int32 accumulator

    fault: None (no fault) OR dict with:
      {
        "kind": "act" | "w" | "acc",
        "out_idx": (n, co, ho, wo),   # which output element is affected
        "tap": (ci, kh, kw),          # which multiply-accumulate inside that output
        "bit": int                    # bit index (0..7 for act/w, 0..31 for acc)
      }

    IMPORTANT: This injects the bit-flip on local register-like variables,
               NOT modifying x_i8 or w_i8 stored in memory.
    """
    assert x_i8.dtype == np.int8 and w_i8.dtype == np.int8
    N, Cin, H, W = x_i8.shape
    Cout, Cin2, kH, kW = w_i8.shape
    assert Cin == Cin2

    # pad input (memory is "protected" in your assumptions; we won't flip bits in x_pad itself)
    x_pad = np.pad(
        x_i8,
        ((0,0),(0,0),(padding,padding),(padding,padding)),
        mode="constant",
        constant_values=0
    )

    Hpad, Wpad = x_pad.shape[2], x_pad.shape[3]
    Hout = (Hpad - kH)//stride + 1
    Wout = (Wpad - kW)//stride + 1

    out = np.zeros((N, Cout, Hout, Wout), dtype=np.int32)

    injected = False
    inject_info = None

    # main loops
    for n in range(N):
        for co in range(Cout):
            for ho in range(Hout):
                for wo in range(Wout):
                    h0 = ho * stride
                    w0 = wo * stride

                    acc = np.int32(0)

                    for ci in range(Cin):
                        for kh in range(kH):
                            for kw in range(kW):
                                # load "register" values for this MAC
                                a_i8 = np.int8(x_pad[n, ci, h0 + kh, w0 + kw])
                                b_i8 = np.int8(w_i8[co, ci, kh, kw])

                                # inject SEU on activation register
                                if (fault is not None) and (not injected):
                                    if fault["kind"] == "act" and fault["out_idx"] == (n, co, ho, wo) and fault["tap"] == (ci, kh, kw):
                                        before = a_i8
                                        a_i8 = flip_bit_int8(a_i8, fault["bit"])
                                        injected = True
                                        inject_info = {
                                            "kind": "act",
                                            "out_idx": (n, co, ho, wo),
                                            "tap": (ci, kh, kw),
                                            "bit": fault["bit"],
                                            "before": int(before),
                                            "after": int(a_i8),
                                        }

                                # inject SEU on weight register
                                if (fault is not None) and (not injected):
                                    if fault["kind"] == "w" and fault["out_idx"] == (n, co, ho, wo) and fault["tap"] == (ci, kh, kw):
                                        before = b_i8
                                        b_i8 = flip_bit_int8(b_i8, fault["bit"])
                                        injected = True
                                        inject_info = {
                                            "kind": "w",
                                            "out_idx": (n, co, ho, wo),
                                            "tap": (ci, kh, kw),
                                            "bit": fault["bit"],
                                            "before": int(before),
                                            "after": int(b_i8),
                                        }

                                # compute product in int32
                                prod = np.int32(a_i8) * np.int32(b_i8)
                                acc = np.int32(acc + prod)

                                # inject SEU on accumulator register (partial sum)
                                if (fault is not None) and (not injected):
                                    if fault["kind"] == "acc" and fault["out_idx"] == (n, co, ho, wo) and fault["tap"] == (ci, kh, kw):
                                        before = acc
                                        acc = flip_bit_int32(acc, fault["bit"])
                                        injected = True
                                        inject_info = {
                                            "kind": "acc",
                                            "out_idx": (n, co, ho, wo),
                                            "tap": (ci, kh, kw),
                                            "bit": fault["bit"],
                                            "before": int(before),
                                            "after": int(acc),
                                        }

                    out[n, co, ho, wo] = acc

    return out, inject_info


import numpy as np

def checksum_detector_conv_int8_acc32(
    x_i8: np.ndarray,
    w_i8: np.ndarray,
    y_acc: np.ndarray,
    stride: int = 1,
    padding: int = 1,
    bias_i32: np.ndarray | None = None,
    tol: int = 0,
    method: str = "ours",   # "ours" or "tvlsi"
):
    """
    Checksum-based fault detector for conv (int8 x int8 -> int32 acc).

    method:
      - "ours": sum_shift[ch,kh,kw] = sum(x_pad[ch, kh:kh+Hout, kw:kw+Wout])
      - "tvlsi": sum_shift = in_channel_sum - sub 
    """
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

        elif method.lower() == "tvlsi":
            # TVLSI: in_channel_sum - sub
            in_channel_sum = x_pad[n].astype(np.int64).sum(axis=(1, 2))  # shape (Cin,)

            sub = np.zeros((Cin, kH, kW), dtype=np.int64)

            for ci in range(Cin):
                total = in_channel_sum[ci]
                for kh in range(kH):
                    for kw in range(kW):
                        window_sum = x_pad[n, ci, kh:kh + Hout, kw:kw + Wout].astype(np.int64).sum()
                        sub[ci, kh, kw] = total - window_sum
                        sum_shift[ci, kh, kw] = total - sub[ci, kh, kw]  # = window_sum


        else:
            raise ValueError("method must be 'ours' or 'tvlsi'")

        # input_checksum[co] = sum_{ci,kh,kw} w[co,ci,kh,kw] * sum_shift[ci,kh,kw]
        for co in range(Cout):
            input_checksum[n, co] = np.sum(w64[co] * sum_shift)

    # --- Output checksum ---
    for n in range(N):
        for co in range(Cout):
            out_sum = y_acc[n, co].astype(np.int64).sum()
            if bias_i32 is not None:
                out_sum -= np.int64(bias_i32[co]) * np.int64(Hout) * np.int64(Wout)
            output_checksum[n, co] = out_sum

    diff = input_checksum - output_checksum
    detected = np.abs(diff) > np.int64(tol)

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
    return res, info

# ---------------- Example usage ----------------
if __name__ == "__main__":
    np.random.seed(0)
    N_RUNS = 10
    METHOD = "tvlsi"   # "ours" or "tvlsi"
    total_detected = 0
    fn = 0
    KIND = "acc"  # "act", "w", or "acc"
    if KIND == "acc":
        component = "partial sum (accumulator)"
    elif KIND == "w":
        component = "weight"
    elif KIND == "act":
        component = "input activation"
    
    for run in range(N_RUNS):
        N, Cin, H, W = 1, 3, 5, 5
        Cout, kH, kW = 16, 3, 3
        stride, padding = 1, 1

        x = np.random.randint(-5, 6, size=(N, Cin, H, W), dtype=np.int8)
        w = np.random.randint(-3, 4, size=(Cout, Cin, kH, kW), dtype=np.int8)

        # golden run (no fault)
        y_golden, _ = conv2d_int8_acc32_seu(x, w, stride=stride, padding=padding, fault=None)

        # single SEU during inference (register-level), e.g. accumulator fault
        # create a random fault that targets 1 output element and 1 MAC tap
        # kind can be "act", "w", or "acc"
        fault = random_seu_fault_spec(out_shape=y_golden.shape, Cin=Cin, kH=kH, kW=kW, kind=KIND)

        y_faulty, info = conv2d_int8_acc32_seu(x, w, stride=stride, padding=padding, fault=fault)

        # Injection info
        print(f"\n--- RUN {run + 1} ---")
        print("Fault spec:", fault)
        print("Injected info:", info)
        # Run checksum detector
        det, info = checksum_detector_conv_int8_acc32(x, w, y_faulty, stride=1, padding=1, tol=0, method=METHOD)
        if det:
            total_detected += 1
        else:
            fn += 1

    print(f"\n[SUMMARY] Detected {total_detected} out of {N_RUNS} runs using method '{METHOD}', injection component '{component}'.")
        
