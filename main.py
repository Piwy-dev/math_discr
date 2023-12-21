import numpy as np
import pagerank as pr


def main():
    adjacence_mattrix = np.loadtxt("data/adjacency_matrix.csv", delimiter=',')
    personalization_vector = np.loadtxt("data/personalisation_verctor.csv", delimiter=',')
    linear_result = pr.pageRankLinear(adjacence_mattrix, 0.9, personalization_vector)
    power_result = pr.pageRankPower(adjacence_mattrix, 0.9, personalization_vector)
    print("Linear method: \n" + linear_result)
    print("Power method: \n" + power_result)