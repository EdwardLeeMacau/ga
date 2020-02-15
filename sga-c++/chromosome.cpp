/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 ***************************************************************************/

#include <stdio.h>
#include "global.h"
#include "chromosome.h"

/* 
    Create a chromosome 
*/
Chromosome::Chromosome ()
{
    length = 0;
    gene = NULL;
    evaluated = false;
}

/* 
    Create a chromosome with length N

    @param n_length The length of chromosome
*/
Chromosome::Chromosome (int n_length)
{
    gene = NULL;
    init (n_length);
}


Chromosome::~Chromosome ()
{
    delete[] gene;
}


void Chromosome::init (int n_length)
{
    length = n_length;

    if (gene != NULL)
        delete[] gene;

    gene = new bool[length];
    evaluated = false;
}

/*
    Get the value of gene in specified index.

    @params index Index of the bit
*/
int Chromosome::getVal (int index) const
{
    if (index < 0 || index > length)
        outputErrMsg ("Index overrange in Chromosome::operator[]");

    return (gene[index])? 1:0;
}

/*
    Set the value of gene in specified index.
*/
void Chromosome::setVal (int index, int val)
{
    if (index < 0 || index > length)
        outputErrMsg ("Index overrange in Chromosome::operator[]");

    gene[index] = (val==1)? true:false;
    evaluated = false;
}


double Chromosome::getFitness ()
{
    if (evaluated)
        return fitness;
    else
        return (fitness = evaluate ());
}


bool Chromosome::isEvaluated () const
{
    return evaluated;
}

/*  Evaluation function of the chromosome  */
double Chromosome::evaluate ()
{
    evaluated = true;
    return oneMax ();
}

/* 
    OneMax Function

    @return double
*/
double Chromosome::oneMax () const
{
    double result = 0;

    for (int i = 0; i < length; i++)
        result += gene[i];

    return result;
}


Chromosome & Chromosome::operator= (const Chromosome & c)
{
    int i;

    if (length != c.length) {
        length = c.length;
        delete[] gene;
        init (length);
    }

    evaluated = c.evaluated;
    fitness = c.fitness;

    for (i = 0; i < length; i++)
        gene[i] = c.gene[i];

    return *this;
}


void Chromosome::printf () const
{
    for (int i = 0; i < length; i++)
        ::printf ("%d", gene[i]);
}


int Chromosome::getLength () const {
    return length;
}


double Chromosome::getMaxFitness () const
{
    // For OneMax
    return ((double)length - 1e-6);
}
