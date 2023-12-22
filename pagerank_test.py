import unittest
import pagerank as pr
import numpy as np


class TestPageRank(unittest.TestCase):
    adjacence_matrix = np.genfromtxt("data/adjacency_matrix.csv", delimiter=',', skip_header=1)
    alpha = 0.9

    def test_probality_matrix(self):
        """
        Vérifie que la matrice de probabilité de transition est stochoastique : chaque colonne somme à 1.
        """
        P = pr.probality_matrix(self.adjacence_matrix)
        self.assertTrue(np.allclose(pr.probality_matrix(P).sum(axis=0), np.ones(P.shape[0])), 
                        "La matrice de probabilité de transition n'est pas stochoastique.")
        
    def test_google_matrix(self):
        """
        Vérifie que la matrice Google est stochoastique : chaque colonne somme à 1.
        """
        P = pr.probality_matrix(self.adjacence_matrix)
        self.assertTrue(np.allclose(pr.google_matrix(P, self.alpha).sum(axis=0), np.ones(P.shape[0])), 
                        "La matrice Google n'est pas stochoastique.")
    

if __name__ == '__main__':
    unittest.main()