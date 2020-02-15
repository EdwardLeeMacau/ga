/***************************************************************************
 *   Copyright (C) 2004 by Tian-Li Yu                                      *
 *   tianliyu@cc.ee.ntu.edu.tw                                             *
 ***************************************************************************/

#ifndef _CHROMOSOME_H
#define _CHROMOSOME_H

#include <gmp.h>
#include <fplll/nr/matrix.h>
#include <fplll/fplll.h>

using namespace fplll;

class Chromosome
{
public:
    Chromosome ();
    Chromosome (int n_ell);
    Chromosome (const Chromosome& c);

    ~Chromosome ();

    virtual Chromosome& operator = (const Chromosome &c);
    virtual bool operator == (const Chromosome &c);
    virtual bool operator != (const Chromosome &c);
    friend ostream& operator << (ostream &o, Chromosome &c);

    void init (int n_ell);

    int getVal (int index) const;
    void setVal (int index, int val);

    double getFitness ();

    double oneMax () const;

    bool isEvaluated () const;

    int getLength () const;

    /* Function to Inheritance */
    virtual double evaluate ();
    virtual double getMaxFitness () const;
    virtual bool isConstraint ();
    virtual void printf () const;
    virtual void optimize ();

protected:

    bool*  _gene;             // Pointer of the gene array
    double _fitness;
    int    _length;           // Initial Length
    bool   _evaluated;
};
#endif
