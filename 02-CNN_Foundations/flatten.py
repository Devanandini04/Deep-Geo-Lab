import numpy as np

def flatten_forward(A_prev):
    """
    3D Volume (Batch, Height, Width, Channels) ko 2D Matrix (Batch, Features) mein badalta hai.
    """
    # Dimensions retrieve karo
    (m, n_H, n_W, n_C) = A_prev.shape
    
    # Reshape command:
    # 'm' (Batch size) same rahega.
    # '-1' ka matlab: Baaki sab dimensions ko ek line mein multiply kar do.
    A = A_prev.reshape(m, -1)
    
    return A