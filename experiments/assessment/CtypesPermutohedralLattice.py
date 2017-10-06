# coding=utf-8
"""
Numpy permutohedral lattice wrapper.
"""

import numpy as np
from ctypes import *
import ctypes.util

# permutohedral lattice library
permdll = ctypes.CDLL('./lib/permuto.so', mode=ctypes.RTLD_LOCAL)

# ctypes wrapping functions
c_constr = permdll.PermBuild
c_constr.restype = c_void_p

c_init = permdll.PermInit
c_init.argtypes = [c_void_p, POINTER(c_float), c_int, c_int]

c_compute = permdll.PermCompute
c_compute.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]

c_delete = permdll.PermDelete
c_delete.argtypes = [c_void_p]


class PermutohedralLattice(object):
    """
    This function model a permutohedral lattice that can be used for
    high dimensional gaussian filtering.
    See [1] for details.
    """

    def __init__(self, features):
        """
        Constructor of the permutohedral lattice.
        This is where feature splatting ([1] sec 3.1) occurs.
        
        Parameters
        ----------
        features: ndarray
            features to be splatted. (h, w, cf)
        """
        # build the lattice
        self.obj = c_constr()

        # prepare features for splatting
        h, w, cf = features.shape
        features = np.transpose(features, (2, 0, 1))  # channels first
        features = np.reshape(features, (cf, h*w))
        features = np.ascontiguousarray(features, dtype='float32')

        # splat
        f_p = features.ctypes.data_as(POINTER(c_float))
        c_init(self.obj, f_p, cf, h*w)

        # pre-compute normalization factor (in case we want to normalize the filter)
        # norm_factor is computed by filtering ones
        self.norm_factor = self.compute(np.ones((h, w, 1)), normalize=False)

    def compute(self, x, normalize=True):
        """
        Function that performs the filtering.
        Blurring and slicing are applied here ([1] sec 3.1).
        
        Parameters
        ----------
        x: ndarray
            tensor to be filtered. (h, w, cx).
        normalize: bool
            whether or not to normalize the filter (default is True).

        Returns
        -------
        ndarray
            the filtered tensor. (h, w, cx).

        """

        # prepare x
        h, w, cx = x.shape
        x = np.transpose(x, (2, 0, 1))
        x = np.reshape(x, (cx, h*w))  # channels first
        x = np.ascontiguousarray(x, dtype='float32')
        x_p = x.ctypes.data_as(POINTER(c_float))

        # prepare output tensor
        out = np.ones_like(x)
        out = np.ascontiguousarray(out, dtype='float32')
        out_p = out.ctypes.data_as(POINTER(c_float))

        # filtering.
        c_compute(self.obj, x_p, out_p, cx, h*w)

        # reshape output tensor
        out = np.reshape(out, (cx, h, w))
        out = np.transpose(out, axes=(1, 2, 0))

        # optionally normalize the output filter
        if normalize:
            out /= self.norm_factor

        return out

    def compute_stacked(self, x_list, normalize=True):
        """
        Helper function to compute the filtering of more
        than one tensor without calling the compute many times.
        Tensors must have the same dimension (except for the channels)
        Tensors are stacked along channels and filtered as one.
        
        TODO: Not tested yet!
        
        Parameters
        ----------
        x_list: list
            list of ndarrays having shape (h, w, whatever)
        normalize: bool
            whether or not to normalize the filter (default is True).

        Returns
        -------
        list
            list of filtered tensors. (h, w, whatever) 
        """
        x_list = [np.transpose(x, (2, 0, 1)) for x in x_list]
        c_vec = [x.shape[0] for x in x_list]

        # stack as one
        stacked_tensor = np.concatenate(x_list, axis=0)

        # filter as one
        stacked_tensor_filtered = self.compute(stacked_tensor, normalize)

        output_list = []
        c_counter = 0
        for i in range(0, len(c_vec)):
            output_list.append(stacked_tensor_filtered[c_counter:c_counter + c_vec[i]])
            c_counter += c_vec[i]

        return output_list

    def __del__(self):
        """
        Destructor function.
        Frees the memory allocated on heap.
        
        Returns
        -------
        None
        
        """
        c_delete(self.obj)

"""
References
----------

[1] Adams, Andrew, Jongmin Baek, and Myers Abraham Davis. 
"Fast High‐Dimensional Filtering Using the Permutohedral Lattice" 
Computer Graphics Forum. Vol. 29. No. 2. Blackwell Publishing Ltd, 2010.
"""
