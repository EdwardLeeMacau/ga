/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 ***************************************************************************/

#include <stdio.h>
#include "global.h"
#include "chromosome.h"

/*
 **********************************************************
 * class Chromosome                                       *
 **********************************************************
*/

Chromosome::Chromosome ()
{
    _length    = 0;
    _gene      = NULL;
    _evaluated = false;
}

/* 
    Create a chromosome with length N

    @param n_length 
        The length of chromosome
*/
Chromosome::Chromosome (int n_length)
{
    _gene = NULL;
    init (n_length);
}

/* 
    Create a chromosome with copy constructor

    @param n_length 
        The length of chromosome
*/
Chromosome::Chromosome (const Chromosome& c)
{
    _gene = new bool[c._length];

    for (int i = 0; i < c._length; i++)
        _gene[i] = c._gene[i];

    _fitness   = c._fitness;
    _length    = c._length;
    _evaluated = c._evaluated;
}


Chromosome::~Chromosome ()
{
    delete[] _gene;
}

/* 
    Initialize a chromosome with length N

    @param n_length 
        The length of chromosome
*/
void 
Chromosome::init (int n_length)
{
    _length = n_length;

    if (_gene != NULL)
        delete[] _gene;

    _gene = new bool[_length];
    _evaluated = false;
}

/*
    Get the value of gene in specified index.

    @params index 
        Index of the bit
*/
int 
Chromosome::getVal (int index) const
{
    if (index < 0 || index > _length)
        outputErrMsg ("Index overrange in Chromosome::operator[]");

    return (_gene[index])? 1:0;
}

/*
    Set the value of gene in specified index.

    @params index
        Index of the bit
    @params val
        Value to assign
*/
void 
Chromosome::setVal (int index, int val)
{
    if (index < 0 || index > _length)
        outputErrMsg ("Index overrange in Chromosome::operator[]");

    _gene[index] = (val==1)? true:false;
    _evaluated = false;
}


double 
Chromosome::getFitness ()
{
    return (_evaluated)? _fitness : (_fitness = evaluate());
}


bool 
Chromosome::isEvaluated () const
{
    return _evaluated;
}

/* Virtual: Evaluation function of the chromosome */
double 
Chromosome::evaluate ()
{
    return oneMax ();
}

double 
Chromosome::oneMax () const
{
    double result = 0;
    
    for (int i = 0; i < _length; ++i)
        result += _gene[i];
    
    return result;
}

Chromosome& 
Chromosome::operator = (const Chromosome & c)
{
    if (_length != c._length) 
    {
        _length = c._length;    
        delete[] _gene;
        init (_length);
    }

    _evaluated = c._evaluated;
    _fitness = c._fitness;

    for (int i = 0; i < _length; i++)
        _gene[i] = c._gene[i];

    return (*this);
}

bool
Chromosome::operator == (const Chromosome &c)
{
    if (_length != c._length) 
        return false;

    for (int i = 0; i < _length; ++i) 
    { 
        if (_gene[i] != c._gene[i]) 
            return false; 
    }

    return true;
}

bool Chromosome::operator != (const Chromosome &c)
{
    return !operator==(c);
}

int 
Chromosome::getLength () const 
{
    return _length;
}


double 
Chromosome::getMaxFitness () const
{
    // For OneMax
    return ((double)_length - 1e-6);
}

void
Chromosome::printf() const
{
    for (int i = 0; i < _length; ++i)
        ::printf("%d", _gene[i]);
}

bool
Chromosome::isConstraint()
{
    return false;
}

void
Chromosome::optimize()
{
    /* Do Nothing */
}

