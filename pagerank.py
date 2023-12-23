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
    P = probality_matrix(A)

    x = np.linalg.solve((I - alpha * P).T, (1 - alpha) * v)
    return x


def probality_matrix(A: np.matrix) -> np.matrix:
    """
    `A` : np.matrix : matrice d'adjacence

    Retourne la matrice de probabilité de transition P.
    """
    n = A.shape[0]
    P = np.zeros((n, n))

    # Assignation de p_ij = w_ij / Lj : la probabilité de passer de j à i est égale au poids du lien de j à i divisé par le nombre de liens sortants de j
    for i in range(n):
        for j in range(n):
            if A[i,j] != 0:
                Lj = A[i,:].sum() # Nombre de liens sortants de j
                P[i,j] = A[i,j] / Lj

    return P


def google_matrix(P: np.matrix, alpha: float, v: np.array) -> np.matrix:
    """
    `P` : np.matrix : matrice de probabilité de transition
    `alpha` : float : paramètre de téléportation (entre 0 et 1)
    `v` : np.array : vecteur de probabilité initiale

    Retourne la matrice Google G.
    """
    n = P.shape[0] # Nombre de noeuds

    # Créetion du vecteur colonne e = (1, 1, ..., 1), de taille n
    e = np.ones(n).reshape(n, 1)

    # Transposition du vecteur de probabilité initiale v
    vT = v.reshape(1, n) 

    # Calcul de la matrice Google G
    G = alpha * P + (1 - alpha) * e @ vT

    return G


def pageRankPower (A : np.matrix, alpha : float, v : np.array ) -> np.array:
    """
    `A` : np.matrix : matrice d'adjacence
    `alpha` : float : paramètre de téléportation (entre 0 et 1)
    `v` : np.array : vecteur de probabilité initiale

    Retourne le vecteur de probabilité stationnaire du PageRank -> Un vecteur `x` contenant les scores d’importance des noeuds ordonnés dans
    le même ordre que les lignes de la matrice d’adjacence (représentant les noeuds).
    """
    print(A)

    # Calcul de la matrice de probabilité de transition P
    P = probality_matrix(A)
    print(P)

    # Calcul de la matrice Google G
    G = google_matrix(P, alpha, v)
    print(G)

    # Initialisation du vecteur de probabilité stationnaire x par le degré entrant de chaque noeud
    x = np.array([A[:,i].sum() for i in range(A.shape[0])])
    x = x / x.sum()

    # Calcul du vecteur de probabilité stationnaire x
    for i in range(100):
        # Affiche les trois premières itérations
        if i < 3:
            print(x)
        x = x @ G

    return x