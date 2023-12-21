import numpy as np
import pagerank as pr


def main():
    adjacence_mattrix = np.genfromtxt("data/adjacency_matrix.csv", delimiter=',', skip_header=1)
    personalization_vector = np.genfromtxt("data/personalisation_vertor.csv", delimiter=',', skip_header=1)
    linear_result = pr.pageRankLinear(adjacence_mattrix, 0.9, personalization_vector)
    power_result = pr.pageRankPower(adjacence_mattrix, 0.9, personalization_vector)
   #print(linear_result)
    print(power_result)


if __name__ == '__main__':
    main()