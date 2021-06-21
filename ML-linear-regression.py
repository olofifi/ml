import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
import pandas as pd

from numpy.core.getlimits import _register_known_types

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.2 * x, x)

def square(x: np.ndarray) -> np.ndarray:
    return np.power(x,2)

def cube(x: np.ndarray) -> np.ndarray:
    return np.power(x,3)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+ np.exp(-x))

def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)

def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

def arc(x: np.ndarray) -> np.ndarray:
    return np.arctan(x)

def log(x: np.ndarray) -> np.ndarray:
    return np.log(x)

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.01) -> np.ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]

def deriv_chain(chain: Chain,
           x: np.ndarray) -> np.ndarray:
        
        f=[]
        d=[]
        df=1

        f.append(chain[0](x))
        d.append(deriv(chain[0], x))
        newchain = np.delete(chain, 0 , 0)

        i = 0

        for func in newchain:
            f.append(func(f[i]))
            d.append(deriv(func, f[i]))
            i=i+1

        for der in d:
            df=df*der

        return df

def nested(chain: Chain,
           x: np.ndarray) -> np.ndarray:

        f = chain[0](x)

        newchain = np.delete(chain, 0, 0)

        for func in newchain:
            f = func(f)

        return f



x = np.arange(-3, 3, 0.01)

chain = [leaky_relu, square, sigmoid]

f = nested(chain, x)
d = deriv_chain(chain, x)

plt.plot(x, f, 'b-')
plt.plot(x, d, 'r-')
plt.show()

        



