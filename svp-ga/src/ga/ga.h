/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 ***************************************************************************/

#ifndef GA_H
#define GA_H

// #include "chromosome.h"
#include "svpchromosome.h"
#include "statistics.h"
#include "myrand.h"

template <class Ch>
class SimpleGA
{
public:
    SimpleGA ();
    SimpleGA (int n_ell, int n_nInitial, int n_selectionPressure, double n_pc, double n_pm, int n_maxGen, int n_maxFe);

    ~SimpleGA ();

    void init (int n_ell, int n_nInitial, int n_selectionPressure, double n_pc, double n_pm, int n_maxGen, int n_maxFe);

    void initializePopulation ();
    void evaluate ();
    void selection ();
    void optimize (Ch* /* array */);

    /** tournament selection without replacement */
    void tournamentSelection ();

    /** Roulette wheel selection */
    void rwSelection ();

    void crossover ();
    void pairwiseXO (const Ch &, const Ch &, Ch &, Ch &);
    void onePointXO (const Ch &, const Ch &, Ch &, Ch &);
    void uniformXO (const Ch &, const Ch &, Ch &, Ch &, double);
    void populationXO (const Ch *, Ch *, int * , double);
    
    void mutation ();
    void simpleMutation ();
    void mutationClock ();

    void replacePopulation ();

    void showStatistics ();
    void oneRun (bool output = true);
    int doIt (bool output = true);

    bool shouldTerminate ();
    int getNextPopulation ();

    void setOptimalSolution (const Ch& /* ch */ );
    const Ch& getOptimalSolution ();

    Statistics stFitness;       // Fitness Statistics (Use to calculate fitness
                                // for each generation.

protected:

    int _ell;                   // Chromosome length
    int nInitial;               // initial population size
    int nCurrent;               // current population size
    int nNextGeneration;        // population size for the next generation
    int selectionPressure;      // selection pressure

    double pc;                  // prob of XO
    double pm;                  // prob of Mutation
    Ch *population;             // Array of the Chromsome
    Ch *offspring;              // Array of offspring
    int *selectionIndex;        // Array of shuffle indexs.
    int maxGen;
    int maxFe;
    int repeat;
    int fe;
    int generation;
    int bestIndex;

    Ch optimalSolution;
};

/*
template <class Ch>
class GA : public SimpleGA<Ch>
{
public:

    using SimpleGA<Ch>::SimpleGA;
    using SimpleGA<Ch>::init;

    void oneRun (bool output = true);        // Overwrite SimpleGA::OneRun()
    int doIt (bool output = true);           // Overwrite SimpleGA::doIt()

    using SimpleGA<Ch>::selection;
    using SimpleGA<Ch>::crossover;
    using SimpleGA<Ch>::mutation;
    using SimpleGA<Ch>::replacePopulation;
    using SimpleGA<Ch>::shouldTerminate;
    using SimpleGA<Ch>::showStatistics;
    using SimpleGA<Ch>::stFitness;

protected:

    void optimize ();

    using SimpleGA<Ch>::nCurrent;

    using SimpleGA<Ch>::population;
    using SimpleGA<Ch>::offspring;
    using SimpleGA<Ch>::generation;
    using SimpleGA<Ch>::bestIndex;

    using SimpleGA<Ch>::optimalSolution;
};
*/
#endif /* ga.h */
