/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 *                                                                         *
 *   You can redistribute it and/or modify it as you like                  *
 ***************************************************************************/

#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdlib>

#include "statistics.h"
#include "ga.h"
#include "chromosome.h"
#include "global.h"

using namespace std;

int main (int argc, char *argv[])
{

    if (argc != 9) {
        printf ("GA ell nInitial selectionPressure pc pm maxGen maxFe repeat\n");
        return -1;
    }

    int ell               = atoi (argv[1]);     // problem size
    int nInitial          = atoi (argv[2]);     // initial population size
    int selectionPressure = atoi (argv[3]);     // selection pressure
    double pc             = atof (argv[4]);     // pc (Probability of XO)
    double pm             = atof (argv[5]);     // pm (Probability of Mutation)
    int maxGen            = atoi (argv[6]);     // max generation
    int maxFe             = atoi (argv[7]);     // max fe (failure)
    int repeat            = atoi (argv[8]);     // how many time to repeat
                                                // TODO: Select XO Type by arguments.

    Statistics stGenS, stGenF;                  // Statistics of (GenS, GenF)...
                                                // GenS: Generation Successful
                                                // GenF: Generation Failure
    int usedGen;

    printf("Generation,Fitness,Variance\n");

    for (int i = 0; i < repeat; i++) {

        GA ga(ell, nInitial, selectionPressure, pc, pm, maxGen, maxFe);

        if (repeat == 1) {
            usedGen = ga.doIt (true);
        } else {
            usedGen = ga.doIt (false);
        }

        Chromosome ch(ell);
        if (ga.stFitness.getMax() < ch.getMaxFitness()) {
            // printf ("-");
            stGenF.record (usedGen);
        }
        else {
            // printf ("+");
            stGenS.record (usedGen);
        }

        fflush (NULL);

    }

    // printf ("\nAverage Gen of Success: %f\n", stGenS.getMean());
    // printf ("Average Gen of Failure: %f\n", stGenF.getMean());
    // printf ("Total number of Failure: %ld\n", stGenF.getNumber());

    return EXIT_SUCCESS;
}
