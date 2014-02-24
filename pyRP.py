# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:25:25 2014

@author: Pau Rodríguez López
"""

import ctypes as ct
import numpy as np
import os


class Alpha(ct.Structure):
    """Ctypes Structure for RP alpha parameter.

    Fields:
        data: A pointer to an array of doubles.
        size: The number of elements in the array.
    """
    _fields_ = [("size", ct.c_uint), ("data", ct.POINTER(ct.c_double))]


class PyImage(ct.Structure):
    """Ctypes Structure for rgbI in RP.

    Fields:
        rows: Number of rows of the image.
        columns: Number of columns of the image.
        channels: Number of channels of the image.
        data: Pointer to the flattened data of the image.
    """
    _fields_ = [("rows", ct.c_uint), ("columns", ct.c_uint),
                ("channels", ct.c_uint), ("data", ct.POINTER(ct.c_uint8))]


class Proposals(ct.Structure):
    """Ctypes Structure for RP return boxes.

    Fields:
        nProposals: The number of proposals returned.
        proposals: The array of boxes returned by RP.
    """

    _fields_ = [("nProposals", ct.c_uint),
                ("proposals", ct.POINTER(ct.c_double))]


class SpParams(ct.Structure):
    """Ctypes Structure for RP SpParams"""

    _fields_ = [("sigma_", ct.c_double), ("c_", ct.c_double),
                ("min_size_", ct.c_double)]


class FWeights(ct.Structure):
    """Ctypes Structure for RP simWeights parameter"""

    _fields_ = [("wBias_", ct.c_double), ("wCommonBorder_", ct.c_double),
                ("wLABColorHist_", ct.c_double), ("wSizePer_", ct.c_double)]


class RP:
    """Wrapper for Randomized Prim's Algorithm  using ctypes.

    Attributes:
        rp: The ctypes loaded C library.
        params: The dictionary parameters for the execution of RP.

    Methods:
        loadParamsFromNumpy: Loads the params attributes using a pre-saved
        numpy file.
        getProposals: Calls the C++ library to get the boxes.
    """

    def __init__(self):
        self.params = None

        # Loading of the C++ library with ctypes
        self.rp = ct.cdll.LoadLibrary(os.path.join(os.getcwd(), "rp.so"))
        self.rp.pyRP.restype = Proposals
        self.rp.deallocate.restype = None
        self.rp.pyRP.argtypes = [ct.POINTER(PyImage), SpParams, FWeights,
                                 ct.c_uint, ct.c_uint, ct.POINTER(Alpha),
                                 ct.c_int32, ct.c_bool]
        self.rp.deallocate.argtypes = [ct.POINTER(ct.c_double)]

    def loadParamsFromNumpy(self, npyFile):
        self.params = np.load(npyFile).item()

    def getProposals(self, img, params=None):
        if params is not None:
            self.params = params

        if img.shape[2] != 3:
            raise Exception("Three dimensions expected")

        # Load the image into the Ctypes structure:
        pyImage = PyImage()
        pyImage.rows = img.shape[0]
        pyImage.columns = img.shape[1]
        pyImage.channels = img.shape[2]

        img = np.require(img, dtype=np.uint8, requirements=['A', 'O', 'C'])
        image2 = img.ctypes.data_as(ct.POINTER(ct.c_uint8))
        pyImage.data = image2

        # Set random seed:
        if(self.params['rSeedForRun'] == -1):
            self.params['SeedForRun'] = np.int(np.sum(img) %
                                               np.iinfo(np.int32).max)

        # Convert colorspaces to integers to make it easier in the C++:
        if self.params['colorspace'] == "RGB":
            self.params['colorspace'] = 1
        elif self.params['colorspace'] == "rg":
            self.params['colorspace'] = 2
        elif self.params['colorspace'] == "LAB":
            self.params['colorspace'] = 3
        elif self.params['colorspace'] == "Opponent":
            self.params['colorspace'] = 4
        elif self.params['colorspace'] == "HSV":
            self.params['colorspace'] = 5
        else:
            raise Exception("Unknown colorspace")

        # Fill Ctypes parameter structures.
        sp = SpParams(self.params['superpixels']['sigma'],
                      self.params['superpixels']['c'],
                      self.params['superpixels']['min_size'])

        fw = FWeights(self.params['simWeights']['wBias'],
                      self.params['simWeights']['wCommonBorder'],
                      self.params['simWeights']['wLABColorHist'],
                      self.params['simWeights']['wSizePer'])

        # Convert alpha array to Ctypes.
        alpha = Alpha()
        self.params['alpha'] = np.squeeze(self.params['alpha'])
        alpha.size = self.params['alpha'].shape[0]
        data = self.params['alpha'][:]
        alpha.data = data.ctypes.data_as(ct.POINTER(ct.c_double))

        # Compute proposals:
        self.params['nProposals'] = np.int(self.params['approxFinalNBoxes'] / 0.8)

        Proposals_p = ct.POINTER(Proposals)

        # Actual call to the C++ library
        proposals = Proposals_p(self.rp.pyRP(ct.byref(pyImage),
                                sp,
                                fw,
                                self.params['nProposals'],
                                self.params['colorspace'],
                                ct.byref(alpha),
                                self.params['SeedForRun'],
                                True))

        # Access the pointer to the array of boxes to get results:
        nProposals = proposals[0].nProposals
        boxes = np.ctypeslib.as_array(proposals[0].proposals,
                                      shape=(nProposals, 4))

        # Free C++ allocated memory
        self.rp.deallocate(proposals[0].proposals)

        # Return array of boxes.
        return boxes