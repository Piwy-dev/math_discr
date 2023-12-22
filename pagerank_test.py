import unittest
import pagerank as pr
import main as m
import numpy as np


class TestPageRank(unittest.TestCase):
    def test_probality_matrix(self):
        """
        Vérifie que la matrice de probabilité de transition est stochoastique : chaque colonne somme à 1.
        """
        P = np.genfromtxt("data/adjacency_matrix.csv", delimiter=',', skip_header=1)
        self.assertTrue(np.allclose(pr.probality_matrix(P).sum(axis=0), np.ones(P.shape[0])), 
                        "La matrice de probabilité de transition n'est pas stochoastique.")
    

if __name__ == '__main__':
    unittest.main()