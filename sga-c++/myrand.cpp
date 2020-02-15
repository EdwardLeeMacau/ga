/***************************************************************************
                          myrand.cc  -  description
                             -------------------
    begin                : Sep 24 2001
    copyright            : (C) 2001 by Tian-Li Yu
    email                : tianliyu@cc.ee.ntu.edu.tw
 ***************************************************************************/

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "myrand.h"

#ifdef PI
#undef PI
#endif

#define PI 3.14159265
#define N 624

MyRand::MyRand ()
{

    unsigned long init_key[N];

    srand ((unsigned long) time (NULL));

    for (int i = 0; i < N; i++)
        init_key[i] = (unsigned long) time (NULL) * rand ();

    init_by_array (init_key, N);
}


MyRand::~MyRand ()
{
}

/*
    Return true if random number is lower than parameter **prob**

    @params prob
*/
bool MyRand::flip (double prob)
{
    return (uniform () < prob);
}


/* 
    Return number From [0,1) 

    @return double number From [0, 1)
*/
double
MyRand::uniform ()
{
    return genrand_res53 ();
}


/** From [a,b) */
double
MyRand::uniform (double a, double b)
{
    return uniform () * (b - a) + a;
}


/** Int From [a,b] */
int
MyRand::uniformInt (int a, int b)
{
    return (a + (int) (uniform () * (b - a + 1)));
}


/*
    Generate a random shuffled array of size num, from [a,b] 

    @params array 

    @params num num should less or equal to b - a + 1 

    @parmas a, b the minimum and maximum number to generate
*/
void
MyRand::uniformArray (int *array, int num, int a, int b)
{

    int *base = new int[b - a + 1];
    int i;
    int r;

    for (i = 0; i < b - a + 1; i++)
        base[i] = a + i;

    for (i = 0; i < num; i++) {
        r = uniformInt (0, b - a - i);
        array[i] = base[r];
        base[r] = base[b - a - i];
    }

    delete[] base;
}
