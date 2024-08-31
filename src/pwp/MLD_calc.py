'''
This is an implementation of the Kara et al. (2000) Mixed Layer Depth (MLD) calculation
originally by
Kara, A. B., Rochford, P. A., & Hurlburt, H. E. (2000).
An optimal definition for ocean mixed layer depth.
Journal of Geophysical Research: Oceans,
105(C7), 16803–16821. https://doi.org/10.1029/2000JC900072

Romina Piunno
Department of Physics
University of Toronto
6 July 2020
'''

import numpy as np
import seawater as sw
from scipy.interpolate import interp1d

def is_between(x, a, b):
	'''
	returns true iff x in [a, b]
	:param x: value to check
	:param a: interval boundary
	:param b: interval boundary
	:return: boolean value
	'''

	# make sure a<b
	if b<a:
		a,b = b,a

	return a <= x and x <= b

def mld_calc(s, t, z, deltaT=0.8):
	'''
	Calculates mixed layer depth of an ocean profile based on the algorithm
	describes by Kara et al. (2000)
	:param s: salinity profile (psu) 1D array
	:param t: temperature profile (deg C) 1D array
	:param z: depth (m) 1D array
	:param deltaT: temp change threshold for MLD
	:return: mixed layer depth (m from surface)
	'''
	# find index of 10m depth
	i_10m = np.argmin(np.abs(z-10))
	# compute density profile
	d = sw.dens0(s, t)
	# use temp and density at 10m below surface as reference
	d_ref = d[i_10m]
	T_ref = t[i_10m]

	# compute the density change threshold
	delta_d = sw.dens0(s[i_10m], T_ref+deltaT) - sw.dens0(s[i_10m], T_ref)
	delta_d = np.abs(delta_d)

	'''
	search profile for uniform density region defined by the density difference 
	between adjacent layers < 1/10 * density_threshold 
	iteratively update reference density
	'''
	n = i_10m
	while n < (len(d)-1) and np.abs(d[n]-d[n+1]) < 0.1*delta_d:
		d_ref = d[n]
		n+=1

	'''
	If first layer (below 10m threshold) is strongly stratified, 
	return next layer as MLD
	Note: this should only really be a problem for coarse resolution profiles
	i.e. dz > 10m
	'''
	if n == 0:
		return z[n+1]

	# compute density at base of well-mixed region
	n -=1
	d_b = 0
	if d[n] < d[n + 1]:
		d_b = d_ref - delta_d
	else:
		d_b = d_ref + delta_d

	# need to check if depth range contains d_b
	db_in_profile = False
	while n < (len(d)-1):
		if is_between(d_b, d[n], d[n+1]):
			db_in_profile = True
			break
		else:
			n+=1
	'''
	If no depth range is found such that (h_n, h_n+1) contains d_b,
	then we repeat the search from 10m down looking for a deviation of 
	+/- delta_d from the 10m density
	
	If we do find the depth range then we return depth at which d = d_b
	via linear interpolation
	'''
	if db_in_profile:
		h = interp1d(d[n:n+2], z[n:n+2])
		return h(d_b)
	else:
		n = i_10m
		while n < len(d):
			if np.abs(d[i_10m] - d[n]) > delta_d:
				break
			else:
				n+=1


	'''
	if no depth at which density has changed by +/- density_threshold
	MLD is taken as the bottom of the ocean (last depth value in z)
	'''
	if n == len(d):
		return z[-1]
	else:
		return z[n]



