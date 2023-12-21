import numpy as np


def pageRankLinear (A : np.matrix , alpha : float, v : np.array ) -> np.array:
    """
    `A` : np.matrix : matrice d'adjacence
    `alpha` : float : paramètre de téléportation (entre 0 et 1)
    `v` : np.array : vecteur de probabilité initiale

    Retourne le vecteur de probabilité stationnaire du PageRank -> Un vecteur `x` contenant les scores d’importance des noeuds ordonnés dans
    le même ordre que les lignes de la matrice d’adjacence (représentant les noeuds).
    """
    n = A.shape[0]
    I = np.identity(n)

    # Calculate the transition matrix M
    M = (1 - alpha) * A + alpha / n * np.ones((n, n))

    # Solve the linear system to find the PageRank vector
    x = np.linalg.solve(I - M, v)

    return x

def pageRankPower (A : np.matrix, alpha : float, v : np . array ) -> np.array:
    """
    `A` : np.matrix : matrice d'adjacence
    `alpha` : float : paramètre de téléportation (entre 0 et 1)
    `v` : np.array : vecteur de probabilité initiale

    Retourne le vecteur de probabilité stationnaire du PageRank -> Un vecteur `x` contenant les scores d’importance des noeuds ordonnés dans
    le même ordre que les lignes de la matrice d’adjacence (représentant les noeuds).
    """
    n = A.shape[0]
    B = alpha*A + (1-alpha)*np.ones((n,n))/n
    x = v
    for i in range(100):
        x = B@x
    return x

