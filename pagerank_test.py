import unittest
import pagerank as pr
import main as m
import numpy as np


class TestPageRank(unittest.TestCase):
    def test_linear(self):
        """
        Test the linear method of PageRank.
        """
        A = np.loadtxt("data/adjacency_matrix.csv", delimiter=',')
        alpha = 0.9
        v = np.loadtxt("data/personalization_vector.csv", delimiter=',')
        expected = np.array([]) #TODO : Remplir avec le résultat attendu
        self.assertTrue(np.allclose(pr.pageRankLinear(A, alpha, v), expected))


    def test_power(self):
        """
        Test the power method of PageRank.
        """
        A = np.loadtxt("data/adjacency_matrix.csv", delimiter=',')
        alpha = 0.9
        v = np.loadtxt("data/personalization_vector.csv", delimiter=',')
        expected = np.array([]) #TODO : Remplir avec le résultat attendu
        self.assertTrue(np.allclose(pr.pageRankPower(A, alpha, v), expected))


if __name__ == '__main__':
    unittest.main()