import numpy as np

def tensor_value_info(x, name=None):
    if name is not None:  
        print(f"tensor {name} infomation < shape: {x.shape} dtype: {x.dtype} min: {np.min(x):.5f} max: {np.max(x):.5f} mean: {np.mean(x):.7f} >")
    else:
        print(f"< shape: {x.shape} dtype: {x.dtype} min: {np.min(x):.5f} max: {np.max(x):.5f} mean: {np.mean(x):.7f} >")
