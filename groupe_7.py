import numpy as np

def pageRankLinear (A : np.matrix , alpha : float , v : np.array ) -> np.array:
    n = A.shape[0]
    B = alpha*A + (1-alpha)*np.ones((n,n))/n
    return np.linalg.solve(B,v)

def pageRankPower (A : np.matrix , alpha : float , v : np . array ) -> np.array:
    n = A.shape[0]
    B = alpha*A + (1-alpha)*np.ones((n,n))/n
    x = v
    for i in range(100):
        x = B@x
    return x

