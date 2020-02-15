#ifndef ECGA_H
#define ECGA_H

#include "ga.h"
#include "myrand.h"

template <class Ch>
class cGA : public SimpleGA
{
public:
    cGA();
    ~cGA();
private:
    void sample();
};

template <class Ch>
class ECGA : public SimpleGA
{
public:
    ECGA();
    ~ECGA();

private:
    void buildModel();

    double DL();
    double MDL();
    double DDL();

    double epsilon;

    // Model bbModel;
};

#endif /* ecga.h */
