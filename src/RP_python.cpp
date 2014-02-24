/**
 * RP_python.cpp
 * @abstract Python Ctypes wrapper for Prime Object Proposals with Randomized Prim's Algorithm,
 * Santiago Manen, Matthieu Guillaumin, Luc Van Gool,
 * International Conference on Computer Vision 2013
 * http://www.vision.ee.ethz.ch/~smanenfr/rp/index.html
 *
 * @author Pau Rodríguez
 * @version 1.0 24/02/2014
 *
 */

#include<float.h>
#include <math.h>
#include "rp.h"
#include <iostream>

using namespace std;
extern "C"{

// Ctypes structs
struct Alpha{
    uint size;
    double * data;
};

struct PyImage{
    uint rows;
    uint columns;
    uint channels;
    uchar * data;
};

struct Proposals{
    uint nProposals;
    double * proposals;
};

//    void pyRP(struct PyImage,
//              double, double, double,
//              double, double, double, double,
//              uint, uint, struct Alpha, int, bool);


struct Proposals pyRP(const struct PyImage* image,
          struct Params::SpParams sp, struct Params::FWeights fw,
          uint nProposals, uint colorspace, struct Alpha* alpha, int rSeedForRun, bool verbose)
{
    /**
     *  Computes Prime Object Proposals given an image and parameters.
     */

    //NOSTALGIA: Convert 1d to 2d + channels image
    //const uchar (&image_reint)[image->rows][image->columns][image->channels] = *reinterpret_cast<const uchar (*)[image->rows][image->columns][image->channels]>(image->data);

    // Get image dimensions.
    uint imgSize[3];
    imgSize[0] = image->rows;
    imgSize[1] = image->columns;
    imgSize[2] = image->channels;

    if(image->channels < 3)
    {
        cerr << ("input 1 (rgbI) should have should have 3 channels.") << endl;
    }

    // Create image object.
    const Image rgbI(image->data, std::vector<uint>(imgSize, imgSize + 3), RGB);

    // Create and fill parameters.
    Params params = Params();
    params.setSpParams(sp);
    params.setFWeights(fw);
    params.setNProposals(nProposals);

    // Create alpha vector and fill alpha parameter.
    const std::vector<double> vAlpha(alpha->data, alpha->data + alpha->size);
    params.setAlpha(vAlpha);

    params.setRSeedForRun(rSeedForRun);
    params.setVerbose(verbose);

    // Convert colorspace to enum.
    Colorspace csp;
    switch(colorspace)
    {
        case 1: csp = RGB;
            break;
        case 2: csp = rg;
            break;
        case 3: csp = LAB;
            break;
        case 4: csp = Opponent;
            break;
        case 5: csp = HSV;
            break;
    }
    params.setColorspace(csp);

    // Check number of proposals.
    const uint nProposals_ = params.nProposals();
    assert(nProposals_>0);

    // Create return array.
    double * bbProposals = new double[nProposals_*4];

    // Main computations.
    std::vector<BBox> bbProposalsVector = RP( rgbI, params);

    // Fill return array-
    uint k=0;
    for( k=0; k<bbProposalsVector.size();k++){
      bbProposals[k]=bbProposalsVector.at(k).jMin+1;
      bbProposals[nProposals+k]=bbProposalsVector.at(k).iMin+1;
      bbProposals[nProposals*2+k]=bbProposalsVector.at(k).jMax+1;
      bbProposals[nProposals*3+k]=bbProposalsVector.at(k).iMax+1;
    }

    // Fill structure for ctypes.
    struct Proposals retProposals;
    retProposals.nProposals = nProposals_;
    retProposals.proposals = bbProposals;

    return retProposals;

}

// Memory deallocation.
void deallocate(double * ptr)
{
    delete[] ptr;
}
} // extern "C"
