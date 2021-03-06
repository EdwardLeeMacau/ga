/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <assert.h>

#include "global.h"
#include "statistics.h"
#include "myrand.h"
#include "ga.h"

/*
 **********************************************************
 * template class declaration                             *
 **********************************************************
*/

// template class SimpleGA<Chromosome>;
template class SimpleGA<SVPChromosome<mpz_t, mpfr_t> >;

/*
    For all population execute optimize().

    @params array
        option: {population, offspring}
*/
template <class Ch>
void
SimpleGA<Ch>::optimize (Ch* array)
{
    for (int i = 0; i < nCurrent; ++i) { array[i].optimize(); }
}

/*
 **********************************************************
 * class SimpleGA                                         *
 **********************************************************
*/

template <class Ch>
SimpleGA<Ch>::SimpleGA ()
    : optimalSolution()
{
    _ell = 0;
    nInitial = 0;
    nCurrent = 0;
    fe = 0;

    nNextGeneration = 0;
    maxGen = -1;
    maxFe  = -1;

    population      = NULL;
    offspring       = NULL;
    selectionIndex  = NULL;
}

template <class Ch>
SimpleGA<Ch>::SimpleGA (int n_ell, int n_nInitial, int n_selectionPressure, double n_pc, double n_pm, int n_maxGen, int n_maxFe)
    : optimalSolution()
{
    init (n_ell, n_nInitial, n_selectionPressure, n_pc, n_pm, n_maxGen, n_maxFe);
}

template <class Ch>
SimpleGA<Ch>::~SimpleGA ()
{
    delete[] population;
    delete[] offspring;
    delete[] selectionIndex;
}

/*
    Initialization function of Genetic Algorithm

    @params n_ell

    @params n_nInitial
    
    @params n_selectionPressure
    
    @params n_pc
    
    @params n_pm
    
    @params n_maxGen
    
    @params n_maxFe 
*/
template <class Ch>
void
SimpleGA<Ch>::init (int n_ell, int n_nInitial, int n_selectionPressure, double n_pc, double n_pm, int n_maxGen, int n_maxFe)
{
    _ell = n_ell;

    nInitial = n_nInitial;
    nCurrent = nInitial;
    selectionPressure = n_selectionPressure;
    pc = n_pc;
    pm = n_pm;
    maxGen = n_maxGen;
    maxFe  = n_maxFe;

    population     = new Ch[nInitial];
    offspring      = new Ch[nInitial];
    selectionIndex = new int[nInitial];

    for (int i = 0; i < nInitial; i++) {
        population[i].init (_ell);
        offspring[i].init (_ell);
    }

    initializePopulation ();
}

template <class Ch>
void 
SimpleGA<Ch>::initializePopulation ()
{
    double p = 0.5;

    for (int i = 0; i < nInitial; i++) 
    {
        for (int j = 0; j < _ell; j++) 
        {
            if (myRand.uniform () > p)
                population[i].setVal (j, 1);
            else
                population[i].setVal (j, 0);
        }
    }
}

template <class Ch>
void
SimpleGA<Ch>::setOptimalSolution (const Ch& ch)
{
    optimalSolution = ch;

    return;
}

/*
    For now, assuming fixed population size

    TODO: Extend if iterative population is needed.
*/
template <class Ch>
int 
SimpleGA<Ch>::getNextPopulation ()
{
    return nCurrent;
}

/*
    Unified API of selection
*/
template <class Ch>
void 
SimpleGA<Ch>::selection ()
{
    tournamentSelection ();
    //rwSelection ();
}

/* 
    Roulette wheel selection
    This is a O(n^2) implementation
    You can achieve O(nlogn) by using binary search

    Update GA::selectionIndex
*/
template <class Ch>
void 
SimpleGA<Ch>::rwSelection ()
{
    int i, j;

    // Adjusting population size 
    nNextGeneration = getNextPopulation ();

    if (nNextGeneration != nCurrent) {
        delete[] selectionIndex;
        delete[] offspring;
        selectionIndex = new int[nNextGeneration];
        offspring = new Ch[nNextGeneration];

        for (i = 0; i < nNextGeneration; i++)
            offspring[i].init (_ell);
    }

    double totalFitness = 0.0;
    for (i = 0; i < nCurrent; i++) {
	    totalFitness += population[i].getFitness();
    }

    for (i = 0; i < nNextGeneration; i++) {
        double pointer = totalFitness * myRand.uniform();
        int index = -1;
        double partialSum = 0.0;

        for (j = 0; j < nCurrent; j++) {
            partialSum += population[j].getFitness();
                if (partialSum >= pointer) {
                    index = j;
                    break;
                }
        }
        if (index == -1) index = nCurrent - 1;

        selectionIndex[i] = index;
    }
}

/*
    TournamentSelection without replacement.

    Update GA::selectionIndex
*/
template <class Ch>
void 
SimpleGA<Ch>::tournamentSelection ()
{
    int i, j;

    // Adjusting population size 
    nNextGeneration = getNextPopulation ();

    if (nNextGeneration != nCurrent) {
        delete[] selectionIndex;
        delete[] offspring;
        selectionIndex = new int[nNextGeneration];
        offspring = new Ch[nNextGeneration];

        for (i = 0; i < nNextGeneration; i++)
            offspring[i].init (_ell);
    }

    int randArray[selectionPressure * nNextGeneration];

    int q = (selectionPressure * nNextGeneration) / nCurrent;
    int r = (selectionPressure * nNextGeneration) % nCurrent;

    for (i = 0; i < q; i++) {
        myRand.uniformArray (randArray + (i * nCurrent), nCurrent, 0, nCurrent - 1);
    }

    myRand.uniformArray (randArray + (q * nCurrent), r, 0, nCurrent - 1);

    for (i = 0; i < nNextGeneration; i++) {

        int winner = 0;
        double winnerFitness = -DBL_MAX;

        for (j = 0; j < selectionPressure; j++) {
            int challenger = randArray[selectionPressure * i + j];
            double challengerFitness = population[challenger].getFitness ();

            if (challengerFitness > winnerFitness) {
                winner = challenger;
                winnerFitness = challengerFitness;
            }

        }
        selectionIndex[i] = winner;
    }
}

template <class Ch>
void 
SimpleGA<Ch>::crossover ()
{

    // Pairwise XO Scenario
    
    if ((nNextGeneration & 0x1) == 0) 
    { 
        for (int i = 0; i < nNextGeneration; i += 2) {
            pairwiseXO (population[selectionIndex[i]], population[selectionIndex[i + 1]],
                offspring[i], offspring[i + 1]);
        }
    } 
    else 
    {
        for (int i = 0; i < nNextGeneration - 1; i += 2) {
            pairwiseXO (population[selectionIndex[i]], population[selectionIndex[i + 1]],
                offspring[i], offspring[i + 1]);
        }

        offspring[nNextGeneration - 1] = population[selectionIndex[nNextGeneration - 1]];
    }


    // Population XO Scenario
    // populationXO(population, offspring, selectionIndex, pc);

}

template <class Ch>
void 
SimpleGA<Ch>::pairwiseXO (const Ch & p1, const Ch & p2, Ch & c1, Ch & c2)
{
    do
    {
        if (myRand.uniform () < pc) {
            // onePointXO (p1, p2, c1, c2);
            uniformXO (p1, p2, c1, c2, 0.5);
        }
        else {
            c1 = p1;
            c2 = p2;
        }
    } while (c1.isConstraint() || c2.isConstraint());
}

template <class Ch>
void 
SimpleGA<Ch>::onePointXO (const Ch & p1, const Ch & p2, Ch & c1, Ch & c2)
{
    int i;
    int crossSite = myRand.uniformInt(1, _ell-1);

    for (i = 0; i < crossSite; i++) {
        c1.setVal (i, p1.getVal(i));
        c2.setVal (i, p2.getVal(i));
    }

    for (i = crossSite; i < _ell; i++) {
        c1.setVal (i, p2.getVal(i));
        c2.setVal (i, p1.getVal(i));
    }
}

template <class Ch>
void 
SimpleGA<Ch>::uniformXO (const Ch & p1, const Ch & p2, Ch & c1, Ch & c2, double prob)
{
    for (int i = 0; i < _ell; i++) 
    {
        if (myRand.flip (prob)) {
            c1.setVal (i, p1.getVal(i));
            c2.setVal (i, p2.getVal(i));
        }
        else {
            c1.setVal (i, p2.getVal(i));
            c2.setVal (i, p1.getVal(i));
        }
    }
}

/*
    Manipulate the chromosomes using population-wise shuffled

    TODO: 1. Extend this function with different type crossover.
          2. Take care about the chromosomes which meets Constraint.

    @params p 
        The parent chromosomes array
    @params c 
        The children chromosomes array
    @params indexArray
        The array of random indice
    @params prob
        The probability to XO
    
*/
template <class Ch>
void 
SimpleGA<Ch>::populationXO(const Ch* p, Ch* c, int* indexArray, double prob)
{
    int randArray[nCurrent];      // Random Shuffled Int Array

    if (myRand.flip (prob))
    {
        for (int i = 0; i < _ell; ++i) {
            myRand.uniformArray(randArray, nCurrent, 0, nCurrent - 1);

            for (int j = 0; j < nCurrent; ++j) {
                c[j].setVal(i, p[indexArray[randArray[j]]].getVal(i));
            }
        }
    }
}

template <class Ch>
void 
SimpleGA<Ch>::mutation ()
{
    // simpleMutation ();
    mutationClock ();
}

template <class Ch>
void 
SimpleGA<Ch>::simpleMutation ()
{
    for (int i = 0; i < nNextGeneration; i++) {
        for (int j = 0; j < _ell; j++) {
            if (myRand.flip(pm)) {
                int val = offspring[i].getVal(j);
                offspring[i].setVal(j, 1-val);
            }
        }
    }
}

template <class Ch>
void 
SimpleGA<Ch>::mutationClock ()
{
    if (pm <= 1e-6) return; // can't deal with too small pm

    int pointer = (int) (log(1-myRand.uniform()) / log(1-pm) + 1);
    int q, r, val;

    while (pointer < nNextGeneration * _ell) {
        q = pointer / _ell;
        r = pointer % _ell;

        val = offspring[q].getVal(r);
        offspring[q].setVal(r, 1-val);

        // Compute next mutation clock
        pointer += (int) (log(1-myRand.uniform()) / log(1-pm) + 1);
    };
}

template <class Ch>
void 
SimpleGA<Ch>::showStatistics ()
{
    printf ("Gen:%d  Fitness:(Max/Mean/Min):%f/%f/%f Chromsome Length:%d\n",
        generation, stFitness.getMax (), stFitness.getMean (),
        stFitness.getMin (), population[0].getLength ()
    );

    if (!optimalSolution.isEvaluated()) { optimalSolution.evaluate(); } 

    printf ("best chromosome:");
    optimalSolution.printf();
    // population[bestIndex].printf ();
    printf ("\n");
}

template <class Ch>
void 
SimpleGA<Ch>::replacePopulation ()
{
    if (nNextGeneration != nCurrent) {
        delete[] population;
        population = new Ch[nNextGeneration];
    }

    for (int i = 0; i < nNextGeneration; i++)
        population[i] = offspring[i];

    nCurrent = nNextGeneration;
}

/*
    Run 1 time (Generation += 1)

    @params output If true, print the statistics message
*/

template <class Ch>
void 
SimpleGA<Ch>::oneRun (bool output)
{
    selection ();
    crossover ();
    mutation ();
    replacePopulation ();

    double max = -DBL_MAX;
    stFitness.reset ();

    for (int i = 0; i < nCurrent; i++) {
        double fitness = population[i].getFitness ();

        if (fitness > max) 
        {
            max = fitness;
            bestIndex = i;
        }

        stFitness.record (fitness);
    }

    if (output)
        showStatistics ();

    generation++;
}

/*
    Run a complete GA procedure until teminate

    @return generator 
        the generation spent in
*/
template <class Ch>
int 
SimpleGA<Ch>::doIt (bool output)
{
    generation = 0;

    // optimize(population);

    do 
    {
        oneRun (output);

        // If found the best solution within all generation
        if (population[bestIndex].getFitness() > optimalSolution.getFitness ())
            optimalSolution = population[bestIndex];
    } 
    while (!shouldTerminate());

    return generation;
}


/*
    Return the state of GA

    @return bool 
        true if GA should terminated.
*/
template <class Ch>
bool 
SimpleGA<Ch>::shouldTerminate ()
{
    bool termination = false;
    Ch chromosome = population[0];

    // Reach maximal # of failure
    if (maxFe != -1) {
        if (fe > maxFe)
            termination = true;
    }

    // Reach maximal # of generations
    if (maxGen != -1) {
        if (generation > maxGen)
            termination = true;
    }

    /*
    // The best solution takeover the population
    if (stFitness.getMean() > population[0].getMaxFitness())
        termination = true;
    */

    // If any chromosome takeover the population 
    if (stFitness.getMax() - stFitness.getMin() < 1e-6)
        { cout << "Population takeovered. " << endl; termination = true; }

    // Found a satisfactory solution
    // Modified: Keep Going
    if (stFitness.getMax() >= chromosome.getMaxFitness())
        { cout << "Find an optimal solution. " << endl; termination = true; }

    /*
    // The population loses diversity
    if (stFitness.getMax() - 1e-6 < stFitness.getMean())
	{ cout << "SGA convergence" << endl; termination = true; }
    */

    return termination;
}

template <class Ch>
const Ch&
SimpleGA<Ch>::getOptimalSolution()
{
    return optimalSolution;
}

/*
 **********************************************************
 * class SimpleGA<SVPChromosome<ZT, FT>>                  *
 **********************************************************
*/
