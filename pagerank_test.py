import unittest
import pagerank as pr
import numpy as np


class TestPageRank(unittest.TestCase):
    adjacence_matrix = np.genfromtxt("data/adjacency_matrix.csv", delimiter=',', skip_header=1)
    personalisation_vertor = np.genfromtxt("data/personalisation_vertor.csv", delimiter=',', skip_header=1)
    alpha = 0.9

    def test_probality_matrix(self):
        """
        Vérifie que la matrice de probabilité de transition est stochastique : chaque colonne somme à 1.
        """
        P = pr.probality_matrix(self.adjacence_matrix)
        self.assertTrue(np.allclose(pr.probality_matrix(P).sum(axis=0), np.ones(P.shape[0])), 
                        "La matrice de probabilité de transition n'est pas stochastique.")
        
    def test_google_matrix(self):
        """
        Vérifie que la matrice Google est stochastique : chaque colonne somme à 1.
        """
        P = pr.probality_matrix(self.adjacence_matrix)
        self.assertTrue(np.allclose(pr.google_matrix(P, self.alpha).sum(axis=0), np.ones(P.shape[0])), 
                        "La matrice Google n'est pas stochastique.")
        
    def test_page_rank_linear(self):
        """
        Vérifie que le vecteur de probabilité stationnaire du PageRank est bien calculé : la somme des éléments du vecteur est égale à 1.
        """
        x = pr.pageRankLinear(self.adjacence_matrix, self.alpha, self.personalisation_vertor)
        self.assertTrue(np.allclose(x.sum(), 1), "La méthode pageRankLinear ne retourne pas un vecteur de probabilité stationnaire.")

    def test_page_rank_power(self):
        """
        Vérifie que le vecteur de probabilité stationnaire du PageRank est bien calculé : la somme des éléments du vecteur est égale à 1.
        """
        x = pr.pageRankPower(self.adjacence_matrix, self.alpha, self.personalisation_vertor)
        self.assertTrue(np.allclose(x.sum(), 1), "La méthode pageRankPower ne retourne pas un vecteur de probabilité stationnaire.")

    def test_page_rank_linear_vs_power(self):
        """
        Vérifie que les deux méthodes de calcul du PageRank donnent le même résultat.
        """
        x_linear = pr.pageRankLinear(self.adjacence_matrix, self.alpha, self.personalisation_vertor)
        x_power = pr.pageRankPower(self.adjacence_matrix, self.alpha, self.personalisation_vertor)
        self.assertTrue(np.allclose(x_linear, x_power, atol=0.018), "Les méthodes pageRankLinear et pageRankPower ne donnent pas le même résultat.")
    

if __name__ == '__main__':
    unittest.main()