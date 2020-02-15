#ifndef _SVPCHROMOSOME_H
#define _SVPCHROMOSOME_H
#include "chromosome.h"

/*
 **********************************************************
 * public function                                        *
 **********************************************************
*/

template <class ZT> void HadamardRatio (const ZZ_mat<ZT>& mat);
template <class ZT> bool ascending (const MatrixRow<Z_NR<ZT> >&, const MatrixRow<Z_NR<ZT> >&);
template <class ZT> bool descending (const MatrixRow<Z_NR<ZT> >&, const MatrixRow<Z_NR<ZT> >&);
template <class ZT> void SortMatrix (ZZ_mat<ZT>& mat);

template <class ZT> Z_NR<ZT> L2Norm (const NumVect<Z_NR<ZT> >& vec);
template <class ZT> Z_NR<ZT> L2Norm (const MatrixRow<Z_NR<ZT> >& vec);
template <class FT> FP_NR<FT> L2Norm (const NumVect<FP_NR<FT> >& vec);
template <class FT> FP_NR<FT> L2Norm (const MatrixRow<FP_NR<FT> >& vec);

/*
 **********************************************************
 * class SVPChromosome                                    *
 *     ZT: Type of Int                                    *
 *     FT: Type of floating                               *
 **********************************************************
*/

template <class ZT, class FT>
class SVPChromosome : public Chromosome
{

public:

    #define NEG 1

    SVPChromosome ();
    SVPChromosome (int n_dim, int n_length);

    SVPChromosome& operator = (const SVPChromosome&);
    template <ZT, FT> friend ostream& operator << (ostream &o, SVPChromosome<ZT, FT>& c);

    /* Inheritance Member Function */
    double evaluate ();
    double getMaxFitness () const;
    bool isConstraint ();
    void printf () const;

    void optimize ();

    Z_NR<ZT> norm() const;
    NumVect<Z_NR<ZT> > vector() const;
    void genUnitVector(int /* dim */);
    void print_vector () const;

    int digits() const;

    static void setLatticeBasis (const ZZ_mat<ZT> & /* basis */);
    static void setLatticeBasis (istream & /* i */);
    static void setRefFromDet (const char* /* str */);
    static void setRefFromDet (const Z_NR<ZT>& /* det */);

    static const ZZ_mat<ZT>& getLatticeBasis ();
    static const FP_mat<FT>& getMuMatrix ();
    static const FP_NR<FT>& getMu (int /* i */, int /* j */);
    static const Z_NR<ZT> getDet ();
    static const int& getDimension ();
    static const double& getRefNorm ();

    static NumVect<Z_NR<ZT> > vector(const NumVect<Z_NR<ZT> >& /* coeff */);
    static NumVect<Z_NR<ZT> > transform(const NumVect<Z_NR<ZT> >& /* Y */);

protected:
    
    int getInt (int /* dim */);
    void setInt (int /* i */, int /* dim */);

    NumVect<Z_NR<ZT> > decode () const;
    bool encode (const NumVect<Z_NR<ZT>>& /*vec */);
    
    void transform ();
    void itransform ();
    
    int localSearch_Y();
    int localSearch_X();

    double scaleFitness (double /* val */);

    NumVect<Z_NR<ZT> >   _coeff;            // Vector X
    static int           _dimension;        // Dimension of Problem
    static ZZ_mat<ZT>    _latticeBasis;     // Lattice Basis
    static FP_mat<FT>    _muMatrix;         // Mu Matrix
    static double        _refFitness;       // (Terminate Condition)
    static double        _refNorm;          // (Terminate Condition)
};
#endif
