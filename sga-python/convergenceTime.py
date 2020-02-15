import numpy as np

from math import pi, sqrt, log
from matplotlib import pyplot as plt
from scipy.stats import norm

POPULATION_WISE    = [20.06, 28.16, 34.32, 39.88, 44.26, 48.56, 52.20, 56.02, 59.26, 62.52]
UNIFORM_CROSSOVER  = [22.16, 31.28, 38.32, 44.25, 49.81, 54.42, 58.99, 63.04, 66.96, 70.46]
ONEPOINT_CROSSOVER = [25.56, 38.24, 48.32, 56.99, 65.51, 72.79, 80.02, 86.64, 93.49, 99.91]

def main():
    problemSize = np.arange(50, 550, 50)
    populationSize = necessaryPopulationSize(problemSize)

    plt.plot(problemSize, theoreticalTournament(2, problemSize), marker='o', label='Theoretical')
    plt.plot(problemSize, POPULATION_WISE, label='Population-wise Shuffle')
    plt.plot(problemSize, UNIFORM_CROSSOVER, label='Uniform')
    plt.plot(problemSize, ONEPOINT_CROSSOVER, label='One-point')
    plt.legend(loc=0)
    plt.xlabel('Problem Size')
    plt.ylabel('Generation')
    plt.savefig('./p2-1.png')

def necessaryPopulationSize(problemSize):
    return 4 * problemSize * np.log2(problemSize)

def theoreticalTruncation(s=2, problemSize=np.arange(50, 550, 50)):
    selectionIntensity = s * norm.pdf(norm.ppf(1 - 1/s))

    return convergenceTime(problemSize, selectionIntensity)

def theoreticalTournament(s=2, problemSize=np.arange(50, 550, 50)):
    selectionIntensity = sqrt(2 * (log(s) - log(sqrt(4.14 * log(s)))))
    
    return convergenceTime(problemSize, selectionIntensity)

def convergenceTime(size, selectionIntensity):
    return pi / 2 * np.sqrt(size) / selectionIntensity

if __name__ == "__main__":
    main()