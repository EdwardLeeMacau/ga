/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 *                                                                         *
 *   You can redistribute it and/or modify it as you like                  *
 ***************************************************************************/

#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <sstream>
#include <iostream>
#include <cstdlib>

#include "statistics.h"
#include "ga.h"
#include "svpchromosome.h"
#include "global.h"

#include <gmp.h>
#include <mpfr.h>
#include <fplll/nr/matrix.h>
#include <fplll/gso.h>
#include <fplll/bkz.h>
#include <fplll/fplll.h>

using namespace std;
using namespace fplll;

typedef mpz_t ZT;
typedef mpfr_t FT;
typedef SVPChromosome<mpz_t, mpfr_t> SVP;

template <class ZT>
void
SortMatrix (ZZ_mat<ZT>& mat)
{
    stringstream ss;
    vector<tuple<int, int>> tmpM;

    for (int i = 0; i < mat.get_rows(); ++i) { tmpM.push_back(make_tuple(i, (int)L2Norm(mat[i]).get_si())); }

    sort(tmpM.begin(), tmpM.end(), [](const tuple<int, int>& a, const tuple<int, int>& b) { return get<1>(a) < get<1>(b); });

    ss << '[';
    for (int i = 0; i < mat.get_rows(); ++i)
    {
        ss << mat[get<0>(tmpM[i])];
        if (i != mat.get_rows()) { ss << endl; } else { ss << ']' << endl; }
    }
    mat.read(ss);
}

template <class ZT>
Z_NR<ZT>
L2Norm (const MatrixRow<Z_NR<ZT> >& row)
{
    Z_NR<ZT> norm;
    row.dot_product(norm, row);

    return norm;
}

int 
main (int argc, char *argv[])
{
    if (argc != 9) {
        printf ("SGA nVaildDigit nInitial selectionPressure pc pm maxGen maxFe repeat\n");
        return -1;
    }

    int nVaildDigit       = atoi (argv[1]);        // problem size
    int nInitial          = atoi (argv[2]);        // initial population size
    int selectionPressure = atoi (argv[3]);        // selection pressure
    double pc             = atof (argv[4]);        // pc (Probability of XO)
    double pm             = atof (argv[5]);        // pm (Probability of Mutation)
    int maxGen            = atoi (argv[6]);        // max generation
    int maxFe             = atoi (argv[7]);        // max fe (failure)
    int repeat            = atoi (argv[8]);        // how many time to repeat
                                                   // TODO: Select XO Type by arguments.

    string det_str = "";
    Statistics stGenS, stGenF;                     // Statistics of (GenS, GenF)...
                                                   // GenS: Generation Successful
                                                   // GenF: Generation Failure
    
    SVP bestChromosomes[repeat];                   // Array to store the best chromosome
    double criterion;
    int dimension;
    int usedGen;

    ZZ_mat<mpz_t> basis;                           // Lattice Basis Setting 
    basis.read(cin);
    SortMatrix(basis);
    SVP::setLatticeBasis(basis);

    cin >> det_str;                                // Lattice Determinate Setting
    SVP::setRefFromDet(det_str.c_str());

    dimension = SVP::getDimension();

    SVP problem(dimension, nVaildDigit);           // Find shortest basis
    SVP shortest_basis = problem;

    for (int i = 0; i < dimension; ++i)
    {
        problem.genUnitVector(i);
        if (problem.getFitness() > shortest_basis.getFitness())
            shortest_basis = problem;
    }

    // Criterion of successful: 
    //     Find a vector with norm less than gaussian heuristics 
    //     and shorter than all BKZ reduced basis.
    criterion = (problem.getMaxFitness() < shortest_basis.getFitness())? shortest_basis.getFitness() : problem.getMaxFitness();

    cout << "Basis Norm:  " << pow(shortest_basis.norm().get_d(), 0.5) << endl
         << "Max Fitness: " << problem.getMaxFitness() << endl
         << "Ref Norm:    " << problem.getRefNorm() << endl
         << endl;

    printf ("GA Start: \n"); 

    for (int i = 0; i < repeat; i++) 
    {
        SimpleGA<SVP> ga(nVaildDigit * dimension, nInitial, selectionPressure, pc, pm, maxGen, maxFe); 
        ga.setOptimalSolution(shortest_basis);

        if (repeat == 1) 
            usedGen = ga.doIt (true);
        else 
            usedGen = ga.doIt (false);

        bestChromosomes[i] = ga.getOptimalSolution();

        if (bestChromosomes[i].getFitness() < criterion)
            stGenF.record (usedGen);
        else 
            stGenS.record (usedGen);

        fflush (NULL);
    }

    printf ("\nAverage Gen of Success: %f\n", stGenS.getMean());
    printf ("Average Gen of Failure: %f\n", stGenF.getMean());
    printf ("Total number of Failure: %ld\n", stGenF.getNumber());

    printf ("\n---------------------\n");

    for (int i = 0; i < repeat; ++i)
    {
        printf ("Result found in Experiment.%d\n", i); 
        bestChromosomes[i].print_vector();
        printf ("\n");
    }

    return EXIT_SUCCESS;
}
