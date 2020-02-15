import math
import numpy as np
from matplotlib import pyplot as plt

p_3 = 0.005789
p_4 = 0.000277

nCr = lambda n, r: int(math.factorial(n) / math.factorial(r) / math.factorial(n - r))
FORMULA = lambda p, k, ell: (1 - p) ** nCr(ell, k)

def main():
    ells = list(range(5, 150))
    prob_3 = np.array([ FORMULA(p_3, 3, ell) for ell in ells ])
    prob_4 = np.array([ FORMULA(p_4, 4, ell) for ell in ells ])
    
    # print(ells[np.where(prob_3 < 0.5)[0][0]], prob_3[np.where(prob_3 < 0.5)[0][0]])
    # print(ells[np.where(prob_4 < 0.5)[0][0]], prob_4[np.where(prob_4 < 0.5)[0][0]])
    print(np.where((1 - prob_3) < (1 - prob_4)))

    plt.plot(ells, prob_3, label='k=3')
    plt.plot(ells, prob_4, label='k=4')
    plt.plot(ells, [0.5] * len(ells), linestyle='--')
    plt.legend(loc=0)
    plt.xlabel('Problem Size')
    plt.ylabel('Probability')
    plt.title('Probability of k-deception not occured')
    plt.savefig('./p1-1.png')

    plt.clf()
    plt.plot(ells, prob_3, label='k=3')
    plt.plot(ells, prob_4, label='k=4')
    plt.loglog()
    plt.legend(loc=0)
    plt.xlabel('Problem Size')
    plt.ylabel('Probability')
    plt.title('Probability of k-deception not occured')
    plt.savefig('./p1-2.png')

    return

if __name__ == '__main__':
    main()