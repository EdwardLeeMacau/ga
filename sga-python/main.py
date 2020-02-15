import sga

def main():
    genetic = sga.GeneticAlgorithm(17, 20)

    for _ in range(500):
        genetic.evolve()

    return genetic.result()

if __name__ == "__main__":
    main()