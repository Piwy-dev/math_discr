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

def pageRankPower (A : np.matrix, alpha : float, v : np.array ) -> np.array:
    """
    `A` : np.matrix : matrice d'adjacence
    `alpha` : float : paramètre de téléportation (entre 0 et 1)
    `v` : np.array : vecteur de probabilité initiale

    Retourne le vecteur de probabilité stationnaire du PageRank -> Un vecteur `x` contenant les scores d’importance des noeuds ordonnés dans
    le même ordre que les lignes de la matrice d’adjacence (représentant les noeuds).
    """
    print(A)

    # Création de la matrice de probabilité de transition P
    n = A.shape[0]
    P = np.zeros((n, n))

    # Assignation de p_ij = w_ij / nombre de liens sortants de j
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                Lj = A[:,j].sum() # Nombre de liens sortants de j
                P[i,j] = A[i,j] / Lj

    print(P)

    # fait la somme de toutes les colonnes de M
    # print("M est stochastique : ", np.allclose(M.sum(axis=0), np.ones(n)))

    # Calcul de la matrice Google G
    G = alpha * P + (1 - alpha) * v

    print(G)

    
