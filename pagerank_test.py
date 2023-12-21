import unittest
import pagerank as pr
import main as m
import numpy as np


class TestPageRank(unittest.TestCase):
    def test_linear(self):
        """
        Test the linear method of PageRank.
        """
        A = np.array([[0, 1/2, 1/2], [1/3, 0, 1/2], [1/3, 1/2, 0]])
        alpha = 0.9
        v = np.array([1/3, 1/3, 1/3])
        expected = np.array([0.38709677, 0.29032258, 0.32258065])
        self.assertTrue(np.allclose(pr.pageRankLinear(A, alpha, v), expected))


    def test_power(self):
        """
        Test the power method of PageRank.
        """
        A = np.array([[0, 1/2, 1/2], [1/3, 0, 1/2], [1/3, 1/2, 0]])
        alpha = 0.9
        v = np.array([1/3, 1/3, 1/3])
        expected = np.array([0.38709677, 0.29032258, 0.32258065])
        self.assertTrue(np.allclose(pr.pageRankPower(A, alpha, v), expected))


    def test_main(self):
        """
        Test the main method of PageRank.
        """
        adjacence_csv = "data/adjacency_matrix.csv"
        personalization_csv = "data/personalization_vector.csv"
        expected = np.array([0.38709677, 0.29032258, 0.32258065])
        self.assertTrue(np.allclose(m.main(adjacence_csv, personalization_csv), expected))


if __name__ == '__main__':
    unittest.main()