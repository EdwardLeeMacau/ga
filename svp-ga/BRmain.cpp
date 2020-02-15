/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 *                                                                         *
 *   You can redistribute it and/or modify it as you like                  *
 ***************************************************************************/

#include <cmath>
#include <cstdio>
#include <string>
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

int main (int argc, char *argv[])
{
    if (argc != 9) {
        printf ("BasisReduction nVaildDigit nInitial selectionPressure pc pm maxGen maxFe repeat\n");
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
    
    vector<ZZ_mat<ZT>> reducedBasis;               // container to save the result;
    double ratios[repeat];
    string det_str = "";

    Statistics stGenS, stGenF;                     // Statistics of (GenS, GenF)...
                                                   // GenS: Generation Successful
                                                   // GenF: Generation Failure
    
    int dimension;
    int usedGen;

    ZZ_mat<ZT> basis; basis.read(cin);             // Parameters setting
    SVP::setLatticeBasis(basis);
    SVP::setRefFromDet(basis(0, 0));

    dimension = SVP::getDimension();

    SVP problem(dimension, nVaildDigit);           // Find shortest basis

    // Criterion of successful: 
    //     TODO
    
    // cout << "Init Hadamard Ratio:  " << Hadamard(basis) 
    //      << endl;

    printf ("Basis Reduction Start: \n"); 

    for (int i = 0; i < repeat; i++) 
    {
        SimpleGA<SVP> ga(nVaildDigit * dimension, nInitial, selectionPressure, pc, pm, maxGen, maxFe); 

        if (repeat == 1) 
            usedGen = ga.doIt (true);
        else 
            usedGen = ga.doIt (false);

        // fflush (NULL);
    }

    // printf ("\nAverage Gen of Success: %f\n", stGenS.getMean());
    // printf ("Average Gen of Failure: %f\n", stGenF.getMean());
    // printf ("Total number of Failure: %ld\n", stGenF.getNumber());

    // printf ("\n---------------------\n");

    // for (int i = 0; i < repeat; ++i)
    // {
    //     printf ("Result found in Experiment.%d: Hadamard: %d\n", i, ratios[i]); 
    // }

    return EXIT_SUCCESS;
}
