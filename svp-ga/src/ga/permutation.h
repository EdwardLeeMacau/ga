#ifndef PERMUTATION_H
#define PERMUTATION_H

#include "chromosome.h"
#include "ga.h"

class PermutationChromosome : public Chromosome
{
public:
private:
    void fitness();
    double TSP(/* const Graph& graph */);
    double Schedule();
};

template <class ch>
class PermuatationGA : public SimpleGA
{
public:
private:
    void crossover();
    void selection();
    void mutation();
};

#endif
