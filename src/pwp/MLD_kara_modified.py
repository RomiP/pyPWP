import numpy as np
import seawater as sw
from scipy.interpolate import interp1d

from MLD_calc import is_between

def kara_modified(s, t, z, deltaT=0.8, prevMLD=None):
	'''
	Calculates mixed layer depth of an ocean profile based on the algorithm
	describes by Kara et al. (2000)
	This modified version begins the MLD search either 10m below the
	surface or 500m above where the previous MLD was found
	:param s: salinity profile (psu) 1D array
	:param t: temperature profile (deg C) 1D array
	:param z: depth (m) 1D array
	:param deltaT: temp change threshold for MLD
	:param prevMLD: MLD (m from surface) of previous time step
	:return: mixed layer depth (m from surface)
	'''

	# start the MLD search either 10m below surface, or 500m above previous MLD
	start_depth = 10
	if prevMLD:
		start_depth = prevMLD - 500
		if start_depth < 10:
			start_depth = 10

	# find index of start depth depth
	i_10m = np.argmin(np.abs(z - start_depth))
	# compute density profile
	d = sw.dens0(s, t)
	# use temp and density at start depth as reference
	d_ref = d[i_10m]
	T_ref = t[i_10m]

	# compute the density change threshold
	delta_d = sw.dens0(s[i_10m], T_ref + deltaT) - sw.dens0(s[i_10m], T_ref)
	delta_d = np.abs(delta_d)

	'''
	search profile for uniform density region defined by the density difference 
	between adjacent layers < 1/10 * density_threshold 
	iteratively update reference density
	'''
	n = i_10m
	while n < (len(d) - 1) and np.abs(d[n] - d[n + 1]) < 0.1 * delta_d:
		d_ref = d[n]
		n += 1

	# compute density at base of well-mixed region
	n -= 1
	d_b = 0
	if d[n] < d[n + 1]:
		d_b = d_ref - delta_d
	else:
		d_b = d_ref + delta_d

	# need to check if depth range contains d_b
	db_in_profile = False
	while n < (len(d) - 1):
		if is_between(d_b, d[n], d[n + 1]):
			db_in_profile = True
			break
		else:
			n += 1
	'''
	If no depth range is found such that (h_n, h_n+1) contains d_b,
	then we repeat the search from 10m down looking for a deviation of 
	+/- delta_d from the 10m density

	If we do find the depth range then we return depth at which d = d_b
	via linear interpolation
	'''
	if db_in_profile:
		h = interp1d(d[n:n + 2], z[n:n + 2])
		return h(d_b)
	else:
		n = i_10m
		while n < len(d):
			if np.abs(d[i_10m] - d[n]) > delta_d:
				break
			else:
				n += 1

	'''
	if no depth at which density has changed by +/- density_threshold
	MLD is taken as the bottom of the ocean (last depth value in z)
	'''
	if n == len(d):
		return z[-1]
	else:
		return z[n]

def MLD_gradient_search(s, t, z, deltaT=0.8, prevMLD=None):
	'''
	Calculates mixed layer depth of an ocean profile based a density
	gradient search
	:param s: salinity profile (psu) 1D array
	:param t: temperature profile (deg C) 1D array
	:param z: depth (m) 1D array
	:param deltaT: temp change threshold for MLD
	:param prevMLD: MLD (m from surface) of previous time step
	:return: mixed layer depth (m from surface)
	'''
	# find index of 10m depth
	i_10m = np.argmin(np.abs(z - 10))
	# compute density profile
	d = sw.dens0(s, t)
	# use temp and density at 10m below surface as reference
	d_ref = d[i_10m]
	T_ref = t[i_10m]

	# compute the density change threshold
	delta_d = sw.dens0(s[i_10m], T_ref + deltaT) - sw.dens0(s[i_10m], T_ref)
	delta_d = np.abs(delta_d)

	# find where the density gradient exceeds 10% of the threshold
	d_d = np.diff(d)
	mlds = np.squeeze(np.argwhere(abs(d_d[i_10m:]) > 0.1 * delta_d))
	mlds += i_10m

	# if the whole water column is well mixed, MLD is
	# taken to be the bottom of the column
	try:
		if len(mlds) == 0:
			return z[-1]

	except:
		print('got stuck here')
		return z[-1]

	# if no previous MLD is defined, we take the shallowest instance
	if not prevMLD:
		mld = z[mlds[0]]
		return mld

	mld = z[mlds[0]]
	return mld

	layers = [mlds[0]]
	for i in range(len(mlds) - 1):
		if mlds[i+1] != 1 + mlds[i]:
			layers.append(mlds[i+1])

	mld_min = np.inf
	mld = None
	for i in layers:
		h = z[i]
		diff = abs(h - prevMLD)
		if diff < mld_min:
			mld_min = diff
			mld = h

	return mld

	print()

def MLD_density_diff(s, t, z, deltaD=1e-4, min_depth=10):
	'''
	Calculated the MLD by finding the minimum depth at which the
	density exceeds the reference/surface density by the specified
	amount
	:param s: salinity profile (psu) 1D array
	:param t: temperature profile (deg C) 1D array
	:param z: depth (m) 1D array
	:param deltaD: density difference threshold (kg m^-3)
	:param min_depth: reference depth (m from surface)
	:return: mixed layer depth (m from surface)
	'''

	# compute density profile
	d = sw.dens0(s, t)

	# ignore surface effects by starting some depth below the surface
	# min_depth = 10
	rho0_idx = np.argmin(np.abs(z - min_depth))

	# get reference density
	rho0 = d[rho0_idx]

	# find where density exceeds that of reference by min amount
	mld_idx = np.flatnonzero(d[rho0_idx:] - rho0 > deltaD)

	if mld_idx.size == 0:
		# if no such density is found, choose bottom of profile as MLD
		mld = z[-1]
	else:
		mld = z[mld_idx[0] + rho0_idx]

	return mld