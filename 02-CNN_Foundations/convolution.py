import numpy as np

def zero_pad(X, pad):
    """
    Image ke charo taraf zeros ki boundary lagata hai.
    Taaki corner wale pixels bhi achhe se process ho sakein.
    """
    # X shape: (batch_size, height, width, channels)
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Ek chhota sa hissa (slice) leta hai aur uspar filter (W) apply karta hai.
    Math: Sum(Slice * Filter) + Bias
    """
    # Element-wise multiplication
    s = np.multiply(a_slice_prev, W)
    # Sum of all values
    Z = np.sum(s)
    # Add bias
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Puri image par filter ghumata hai (Forward Pass).
    """
    # Dimensions nikal rahe hain
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Output ke naye dimensions calculate karo
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Output volume initialize karo (zeros ke sath)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Padding lagao
    A_prev_pad = zero_pad(A_prev, pad)
    
    # --- MAIN LOOP (Sliding Window) ---
    for i in range(m):               # Har image ke liye
        a_prev_pad = A_prev_pad[i]
        
        for h in range(n_H):         # Height (Vertical move)
            for w in range(n_W):     # Width (Horizontal move)
                for c in range(n_C): # Har Filter ke liye
                    
                    # Window ke corners dhundo
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Image ka slice kato
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolution operation karo
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
    
    # Cache save karo (Backprop ke liye kaam aayega)
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache