#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:11:22 2018

@author: fr
"""

def split_profile(R, Z):
    """
    Split a profile in two parts to get a single value for each Z.
    """
    # Assumption on bubble upward orientation
    mask_left = R < R[Z.argmin()]
    R_left = R[mask_left]
    Z_left = Z[mask_left]
    R_right = R[~mask_left]
    Z_right = Z[~mask_left]
    return R_left, Z_left, R_right, Z_right