#include <algorithm>
#include <math.h>
#include <tgmath.h>
#include "svpchromosome.h"

/*
 **********************************************************
 * template class declaration                             *
 **********************************************************  
*/

template class SVPChromosome<mpz_t, mpfr_t>;

/*
 **********************************************************
 * static member initialization                           *
 **********************************************************
*/

template <class ZT, class FT>
int SVPChromosome<ZT, FT>::_dimension;

template <class ZT, class FT>
ZZ_mat<ZT> SVPChromosome<ZT, FT>::_latticeBasis;

template <class ZT, class FT>
FP_mat<FT> SVPChromosome<ZT, FT>::_muMatrix;

template <class ZT, class FT>
double SVPChromosome<ZT, FT>::_refFitness;

template <class ZT, class FT>
double SVPChromosome<ZT, FT>::_refNorm;


/*
 **********************************************************
 * public function                                        *
 **********************************************************
*/

template <class ZT>
double
HadamardRatio (const ZZ_mat<ZT>& mat)
{
    // TODO
    return 0.0;
}

template <class ZT>
bool
ascending (const MatrixRow<Z_NR<ZT> >& a, const MatrixRow<Z_NR<ZT> >& b)
{
    // TODO
    return (L2Norm(a) > L2Norm(b));
}

template <class ZT>
bool
descending (const MatrixRow<Z_NR<ZT> >& a, const MatrixRow<Z_NR<ZT> >& b)
{
    // TODO
    return (L2Norm(a) < L2Norm(b));
}

template <class ZT>
bool 
shuffle (const MatrixRow<Z_NR<ZT> >& a, const MatrixRow<Z_NR<ZT> >& b)
{
    // TODO
    return true;
}

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
L2Norm (const NumVect<Z_NR<ZT> >& vec)
{
    Z_NR<ZT> norm; norm = (long)0;

    for (int i = 0; i < vec.size(); i++)
        norm.addmul(vec[i], vec[i]); 

    return norm;
}

template <class FT>
FP_NR<FT>
L2Norm (const NumVect<FP_NR<FT> >& vec)
{
    FP_NR<FT> norm; norm = 0.0;
    return norm;
}

template <class ZT>
Z_NR<ZT>
L2Norm (const MatrixRow<Z_NR<ZT> >& row)
{
    Z_NR<ZT> norm;
    row.dot_product(norm, row);
    return norm;
}

template <class FT>
FP_NR<FT>
L2Norm (const MatrixRow<FP_NR<FT> >& row)
{
    FP_NR<FT> norm;
    row.dot_product(norm, row);
    return norm;
}

template <class ZT>
NumVect<Z_NR<ZT> >
neighbour (const NumVect<Z_NR<ZT> >& vec, int dim, bool sign )
{
    NumVect<Z_NR<ZT> > newVec;
    newVec = vec;
    return newVec;
}

/*
 **********************************************************
 * class SVPChromosome                                    *
 **********************************************************
*/

template <class ZT, class FT>
SVPChromosome<ZT, FT>::SVPChromosome () 
    : Chromosome ()
{
    /* Do Nothing */
}

template <class ZT, class FT>
SVPChromosome<ZT, FT>::SVPChromosome (int n_dim, int n_length) 
    : Chromosome (n_dim * n_length)
{
    /* Do Nothing */
}

template <class ZT, class FT>
SVPChromosome<ZT, FT>&
SVPChromosome<ZT, FT>::operator = (const SVPChromosome& c)
{
    Chromosome::operator= (c);
    _evaluated = false;

    return (*this);
}

// TODO
template <class ZT, class FT>
ostream& 
operator << (ostream &o, const SVPChromosome<ZT, FT>& c)
{
    return o;
}

template <class ZT, class FT>
Z_NR<ZT>
SVPChromosome<ZT, FT>::norm () const
{
    assert(_evaluated);

    return L2Norm(vector());
}

/*
    Return the vector represented by bitString. 
    It is a Linear Combination of Basis

    @return vector
*/
template <class ZT, class FT>
NumVect<Z_NR<ZT> >
SVPChromosome<ZT, FT>::vector() const
{
    assert (_evaluated);

    // Initialized a vector
    NumVect<Z_NR<ZT> > vec;
    vec.gen_zero(_dimension);

    // Linear Combination
    for (int col = 0; col < _dimension; ++col)
    {
        for (int row = 0; row < _dimension; ++row)
        {
            vec[col].addmul(_latticeBasis(row, col), _coeff[row]);
        }
    }
    
    return vec;
}

/*
    @params dim target dimension
*/
template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::genUnitVector(int dim)
{
    // Set all value 0
    for (int i = 0; i < _length; ++i) setVal(i, 0);

    // But only set the least significant bit at @dim with 1
    setVal((dim + 1) * digits() - 1, 1);
}

/* Print out the repersented coeff */
template <class ZT, class FT>
void 
SVPChromosome<ZT, FT>::print_vector () const
{
    assert (_evaluated);

    NumVect< Z_NR <ZT> > vec = vector();

    // Print the Y Vector
    #ifdef SVPDEBUG
    cerr << decode() << endl;
    #endif

    cout << "X:   " << endl << _coeff << endl
         << "SVP: " << endl << vec << endl
         << "NORM^2 = " << L2Norm(vec) << endl;
}

/*
 **********************************************************
 * class SVPChromosome :: Inheritance Member Function     *
 **********************************************************
*/

/*
    Return fitness of this chromosome

    @return fitness
*/
template <class ZT, class FT>
double 
SVPChromosome<ZT, FT>::evaluate() 
{
    Z_NR<ZT> L2Norm;
    NumVect<Z_NR<ZT> > vec(_dimension);

    L2Norm = (long)0;

    // Set _evaluate as true
    _evaluated = true;
    _fitness = 0;

    // Get _coeff and shortest vector
    transform();
    vec = vector();

    // Check if the vector is not a zero vector
    for (int i = 0; i < _dimension; ++i) 
    { 
        if (!_coeff[i].is_zero())
        {
            // Get L2 Norm
            for (int i = 0; i < _dimension; ++i) { L2Norm.addmul(vec[i], vec[i]); }

            // Save the result
            _fitness = pow(_dimension * 100.0, 3) / (double)L2Norm.get_d();

            break;
        }
    }

    return _fitness;
}

/* Terminate Condition */
template <class ZT, class FT>
double 
SVPChromosome<ZT, FT>::getMaxFitness () const
{
    return _refFitness;
}

/*
    @return bool
        True if the elements in vector is all zero
*/
template <class ZT, class FT>
bool
SVPChromosome<ZT, FT>::isConstraint() 
{
    if (!_evaluated) { evaluate(); }

    for (int i = 0; i < _coeff.size(); ++i)
        { if (!_coeff[i].is_zero()) return false; }

    return true;
}

/* Print out the bitString and represented coeff. */
template <class ZT, class FT>
void 
SVPChromosome<ZT, FT>::printf () const
{
    assert (_evaluated);

    // Print BitString
    Chromosome::printf(); ::printf("\n");

    // Print Vector
    print_vector();  
}

/*
    Return the represented length ofd each digit

    @return digit
        The length of each represented length
*/
template <class ZT, class FT>
int 
SVPChromosome<ZT, FT>::digits () const
{
    return (_length / _dimension);
}

/* TODO */
template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::optimize ()
{
    int update;

    #ifdef SVPDEBUG
    int update;
    cout << "Before:  " << decode() << endl;
    update = localSearch_Y();
    cout << "After:   " << decode() << endl
         << "Iterate: " << update << endl;
    #else
    cout << "LocalSearchReduced: " << localSearch_Y() << endl;
    #endif
}

/*
 **********************************************************
 * class SVPChromosome :: Private Member Function         *
 **********************************************************
*/

/* 
    @param dim
        The specific dimension
    @return tmp
        The value encoded in bitString
*/
template <class ZT, class FT>
int
SVPChromosome<ZT, FT>::getInt (int dim)
{
    int tmp, ratio;
    
    ratio = 1; 
    tmp = 0;
    
    // Calculate absolute value of Y[dim]
    for (int i = digits() - 1; i > 0; --i)
    {
        if (_gene[dim * digits() + i]) 
            { tmp += ratio; }
        
        ratio = ratio << 1;
    }

    // If NEG, Set tmp = -tmp;
    if (_gene[dim * digits()] == NEG) 
        { tmp *= -1; } 

    return tmp;
}

/* 
    @param num
        The value to encode
    @param dim
        The dimension where the bits to assign.
*/
template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::setInt (int num, int dim)
{
    bool sign = (num < 0)? true : false;

    // If NEG, Set SIGNBIT to true;
    _gene[dim * digits()] = sign;
    num = abs(num);

    // Set absolute value of Y[dim] from LSB to MSB
    for (int i = digits() - 1; i > 0; --i)
    {
        _gene[dim * digits() + i] = (num % 2)? true: false;
        num = num >> 1;
    }

}

/*
    TODO
    Replace the code with getInt(i)
    Transform BitString to Y

    @return Y
        Vector Y, \vec{Y}.
*/
template <class ZT, class FT>
NumVect<Z_NR<ZT> > 
SVPChromosome<ZT, FT>::decode() const
{
    NumVect<Z_NR<ZT> > Y; 
    Z_NR<ZT> tmp;
    unsigned int ratio;

    Y.gen_zero(_dimension);

    for (int i = 0; i < _dimension; ++i)
    {
        tmp = (long)0;
        ratio = 1;

        // Calculate absolute value of Y
        for (int j = digits() - 1; j > 0; --j)
        {
            if (_gene[i * digits() + j]) { tmp.add_ui(tmp, ratio); }
            ratio = ratio << 1;
        }

        // If NEG, Set tmp = -tmp;
        if (_gene[i * digits()] == NEG) { tmp.neg(tmp); } 

        // Assign Value
        Y[i] = tmp;
    }

    return Y;
}

/*
    Transform Y to BitString

    @params vec
        The vector to be encode
    @return bool
        True if the vector in encodable.
*/
template <class ZT, class FT>
bool
SVPChromosome<ZT, FT>::encode (const NumVect<Z_NR<ZT> >& vec)
{
    int tmp;

    // Chech whether the vector can be represent by BINARY
    // Representation region: Not over 2 ^ (bit - 1) - 1
    // Example: BITS = 3, MAX = 2 ^ (3 - 1) - 1 = 3
    for (int i = 0; i < _dimension; ++i)
    {
        if (abs(vec[i].get_si()) > pow(2, digits() - 1) - 1) 
            { return false; }
    }

    for (int i = 0; i < _dimension; ++i)
        { setInt((int)vec[i].get_si(), i); }

    return true;
}

/* 
    Transform BitString to X given Mu Matrix. 
    Stored the result of X to _coeff
*/
template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::transform ()
{
    NumVect<Z_NR<ZT> > Y;
    Z_NR<ZT> tmpz;
    FP_NR<FT> tmpf, tmpf2;
    
    Y = decode();
    _coeff.gen_zero(_dimension);

    for (int i = _dimension - 1; i >= 0; --i)
    {
        tmpf = (double)0;

        // T = sum((mu(i, j) * x_j) for j from i + 1 to n)
        for (int j = i + 1; j < _dimension; ++j)
        {
            tmpf2.set_z(_coeff[j], GMP_RNDN);
            tmpf.addmul(_muMatrix(j, i), tmpf2, GMP_RNDN);
        }

        // Set X = Y - round(T)
        tmpz.set_f(tmpf);
        _coeff[i].sub(Y[i], tmpz);
    }
}

/*
    TODO:
    Inverse Transform X to BitString given Mu Matrix.
*/
template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::itransform ()
{
    NumVect<Z_NR<ZT> > Y;

    Y.gen_zero(_dimension);

    for (int i = 0; i < _dimension; ++i)
    {
        _coeff;
    }

    encode(Y);
}

/* 
    LocalSearch Algorithm via \vec{Y}

    @return counter
        Number of iterations.
*/
template <class ZT, class FT>
int
SVPChromosome<ZT, FT>::localSearch_Y ()
{
    NumVect<Z_NR<ZT> > Y, nextY, Y_tmp;
    double norm, next_norm, norm_tmp;
    int counter = 0;
    bool enable = false;

    Y = decode();

    while (true)
    {
        norm = L2Norm(vector(transform(Y))).get_d();
        next_norm = norm;

        // For each dimension, execute +1 / -1
        for (int dim = 0; dim < _dimension; ++dim)
        {
            for (int sign = 0; sign < 2; ++sign)
            {
                Y_tmp = Y;

                if (!sign) 
                    { Y_tmp[dim].add_ui(Y_tmp[dim], 1); }
                else 
                    { Y_tmp[dim].sub_ui(Y_tmp[dim], 1); }

                norm_tmp = L2Norm(vector(transform(Y_tmp))).get_d();

                if (norm_tmp < next_norm)
                    { next_norm = norm_tmp; nextY = Y_tmp; }
            }
        }

        // If norm is not less than Norm(Y), Stop LocalSearch
        if (next_norm == norm) { break; }

        // Else try to encode Y as bitstring
        // enable = encode(nextY);
        // if (enable) { evaluate(); }

        // Start next iteration
        Y = nextY;
        ++counter;
    }

    return (encode(nextY))? counter : 0;
}

/* 
    TODO 
    Start LocalSearch Algorithm via \vec{Y}

    @return counter
        Number of iterations.
*/
template <class ZT, class FT>
int
SVPChromosome<ZT, FT>::localSearch_X ()
{
    int counter = 0;
    
    while (true)
    {
        break;
        ++counter;
    }

    return counter;
}

/*
    @param val
        Norm value of the vector
*/
template <class ZT, class FT>
double
SVPChromosome<ZT, FT>::scaleFitness (double val)
{
    return (pow(100 * _dimension, 3) / (val * val) );
}

/*
 **********************************************************
 * class SVPChromosome :: Static Member Function          *
 **********************************************************
*/

template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::setLatticeBasis (istream& i)
{
    ZZ_mat<ZT> basis; basis.read(i);
    setLatticeBasis(basis);
}

template <class ZT, class FT>
void 
SVPChromosome<ZT, FT>::setLatticeBasis (const ZZ_mat<ZT>& basis) 
{ 
    // Get Attritubes
    _latticeBasis = basis;
    _dimension = basis.get_rows();
    _muMatrix.resize(_dimension, _dimension);

    ZZ_mat<ZT> u, u_inv_t;
    MatGSO<Z_NR<ZT>, FP_NR<FT> > gso(_latticeBasis, u, u_inv_t, 0);
    // mpz_t estimatedNorm;

    // Get Mu Matrix
    gso.update_gso();
    
    for (int i = 0; i < _dimension; ++i) 
    { 
        for (int j = 0; j < _dimension; ++j) 
            gso.get_mu(_muMatrix(i, j), i, j); 
    }

    #ifdef SVPDEBUG
    // cerr << "gso.b = " << endl << gso.b << endl;
    // cerr << "_mu = " << endl << _muMatrix << endl;
    // cout << "Reference L2 Norm: " << _refNorm << endl
    //      << "Reference Fitness: " << _refFitness << endl;
    #endif

    return;
}

template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::setRefFromDet (const char* s)
{
    Z_NR<ZT> det; 
    det.set_str(s);
    setRefFromDet(det);
}

template <class ZT, class FT>
void
SVPChromosome<ZT, FT>::setRefFromDet (const Z_NR<ZT>& d)
{
    mpz_t tmp;
    double det;

    mpz_init(tmp);
    mpz_root(tmp, d.get_data(), _dimension);
    det = mpz_get_d(tmp);

    _refNorm    = 1.05 * pow(tgamma(0.5 * (double)_dimension + 1.0), 1.0 / (double)_dimension) / pow(M_PI, 0.5) * det;
    _refFitness = (pow(100 * _dimension, 3) / (_refNorm * _refNorm) );
}

template <class ZT, class FT>
const ZZ_mat<ZT>& 
SVPChromosome<ZT, FT>::getLatticeBasis () 
{ 
    return _latticeBasis; 
}

template <class ZT, class FT>
const FP_mat<FT>& 
SVPChromosome<ZT, FT>::getMuMatrix () 
{ 
    return _muMatrix; 
}

template <class ZT, class FT>
const FP_NR<FT>&
SVPChromosome<ZT, FT>::getMu (int i, int j)
{
    return _muMatrix(i, j);
}

/*
    TODO:

    @return det 
        Determinant of Lattice, det(L)
*/
template <class ZT, class FT>
const Z_NR<ZT>
SVPChromosome<ZT, FT>::getDet ()
{
    ZZ_mat<ZT> reducedBasis;
    Z_NR<ZT> det, tmp2, tmp3;

    det = (long)0;
     
    return det;
}

template <class ZT, class FT>
const int& 
SVPChromosome<ZT, FT>::getDimension () 
{ 
    return _dimension; 
}


/* Ternimate Condition */
template <class ZT, class FT>
const double&
SVPChromosome<ZT, FT>::getRefNorm ()
{
    return _refNorm;
}

/*
    Return the Linear Combination of Basis given coeff. 

    @return vector
        Linear Combination
*/
template <class ZT, class FT>
NumVect<Z_NR<ZT> >
SVPChromosome<ZT, FT>::vector(const NumVect<Z_NR<ZT> >& coeff)
{
    // Initialized a vector
    NumVect<Z_NR<ZT> > vec;
    vec.gen_zero(_dimension);

    // Linear Combination
    for (int col = 0; col < _dimension; ++col)
    {
        for (int row = 0; row < _dimension; ++row)
        {
            vec[col].addmul(_latticeBasis(row, col), coeff[row]);
        }
    }
    
    return vec;
}

/* 
    Transform Y to X given Mu Matrix. 

    @return coeff
        Vector X, \vec(X)    
*/
template <class ZT, class FT>
NumVect<Z_NR<ZT> >
SVPChromosome<ZT, FT>::transform (const NumVect<Z_NR<ZT> >& Y)
{
    NumVect<Z_NR<ZT> > coeff;
    Z_NR<ZT> tmpz;
    FP_NR<FT> tmpf, tmpf2;
    
    coeff.gen_zero(_dimension);

    for (int i = _dimension - 1; i >= 0; --i)
    {
        tmpf = (double)0;

        // T = \sum((mu(i, j) * x_j) for j from i + 1 to n)
        for (int j = i + 1; j < _dimension; ++j)
        {
            tmpf2.set_z(coeff[j], GMP_RNDN);
            tmpf.addmul(_muMatrix(j, i), tmpf2, GMP_RNDN);
        }

        // Set X = Y - round(T)
        tmpz.set_f(tmpf);
        coeff[i].sub(Y[i], tmpz);
    }

    return coeff;
}
