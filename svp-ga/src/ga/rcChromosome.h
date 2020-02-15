#ifndef RCCHROMOSOME_H
#define RCCHROMOSOME_H

#include "chromosome.h"

class RealNumChromosome : public Chromosome
{
public:
    /* Inheritance member function */    
    RealNumChromosome ();
    RealNumChromosome (int n_ell);
    RealNumChromosome (const Chromosome& c);

    ~RealNumChromosome ();

    virtual RealNumChromosome& operator = (const RealNumChromosome &c);
    virtual bool operator == (const RealNumChromosome &c);
    virtual bool operator != (const RealNumChromosome &c);
    friend ostream& operator << (ostream &o, RealNumChromosome &c);

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


private:
    /* Gene */
    double* _gene;
}

class IntChromosome : public RealNumChromosome
{
public:
private:
}

#endif /* rcChromosome.h */
