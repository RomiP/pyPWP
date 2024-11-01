import numpy as np
from pwp.sw_gas_flux import gasflux
from pwp.MLD_calc import mld_calc
from pwp.MLD_kara_modified import *

# global physical constants
cpw = 4183.3 # specific heat of seawater [J/kgC]
g = 9.81 # gravitational acceleration [m/s^2]

def temperature_flux(pwp, n):
	'''
	Assuming two incoming solar radiation wavelengths (penetrating shortwave,
	and non-penetrating longwave), compute the net heat flux. Absorption profile
	has double exponential depth dependence. Net heat flux is computed by
	F = absorption profile - Q_out
	:param pwp: instance of PWP object
	:param n: index of time step
	:return: depth vector of temperature flux in deg C
	'''
	red = 0.62  # fraction of incoming solar radiation assumed to be longwave
	blue = 1 - red  # fraction of incoming solar radiation assumed to be shortwave

	# compute absorption profile shape (long and short wave)
	absrb_prof =  red * np.exp(-pwp.z / pwp.lambda_red) + blue * np.exp(-pwp.z / pwp.lambda_blue)
	# normalize
	absrb_prof /= np.sum(absrb_prof)
	# multiply by incoming forcing
	absrb_prof *= pwp.q_in[n - 1]
	# subtract surface cooling
	absrb_prof[0] -= pwp.q_out[n - 1]

	# compute resulting temperature changes
	h_flux = absrb_prof * pwp.dt / (pwp.dz * pwp.dens[0, n-1] * cpw)

	return h_flux

def salinity_flux(pwp, n):
	'''
	Computes the surface salinity flux based on current salinity
	and freshwater forcing
	:param pwp: instance of PWP object
	:param n: index of time step
	:return: sfurface salinity flux in psu [float]
	'''
	s_flux = pwp.sal[0,n]*pwp.emp[n-1]*pwp.dt/pwp.dz

	return s_flux

def wanninkoff(pwp, n, gas):
	'''
	computes gas flux using Wannikhoff parametrization
	:param pwp: instance of PWP object
	:param n: index of time step
	:param gas: name of trace gas (e.g. 'co2')'
	:return: depth vector of gas flux in kg/m^3
	'''
	ca = vars(pwp)[gas+'_forcing'][n-1]
	cw = vars(pwp)[gas][0, n-1]
	u = pwp.u[n-1]
	v = pwp.v[n-1]
	t = pwp.temp[0,n-1]

	F = gasflux(ca, cw, u , v, t, gas)
	profile = vars(pwp)[gas][:,n-1]
	profile[0] -= F
	return profile

def mld_kara(pwp, n):
	'''
	computes mixed layer depth using Kara method
	:param pwp: instance of PWP object
	:param n: index of time step
	:return: mixed layer depth in meters [float]
	'''
	mld = mld_calc(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh)
	return mld

def mld_kara_mod(pwp, n):
	'''
		computes mixed layer depth using modified Kara method
		:param pwp: instance of PWP object
		:param n: index of time step
		:return: mixed layer depth in meters [float]
		'''
	if n == 0:
		prevMLD = None
	else:
		prevMLD = pwp.mld[n-1]

	mld = kara_modified(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh, prevMLD)
	return mld

def mld_grad_search(pwp, n):
	'''
	computes mixed layer depth using modified Kara method
	:param pwp: instance of PWP object
	:param n: index of time step
	:return: mixed layer depth in meters [float]
	'''
	if n == 0:
		prevMLD = None
	else:
		prevMLD = pwp.mld[n-1]

	mld = MLD_gradient_search(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh, prevMLD)
	return mld

def mld_density_diff_search(pwp, n):

	dens_threshold = 1e-4
	mld = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold)
	return mld

def mld_density_diff_nearest(pwp, n):

	dens_threshold = 1e-4 # kg/m^3
	mld = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold, 10)

	if n == 0:
		return mld

	prev_mld = pwp.mld[n-1]
	min_depth = prev_mld - 50
	if min_depth < 10:
		min_depth = mld + pwp.dz
	mld2 = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold, min_depth)

	if abs(mld - prev_mld) < abs(mld2 - prev_mld):
		return mld
	else:
		# print('returned mld2')
		return mld2

def mld_density_diff_adjusted(pwp, n):
	dens_threshold = 1e-4 # kg/m^3
	mld = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold)

	if n == 0:
		return mld

	prev_mld = pwp.mld[n-1]
	# restratification threshold: don't allow mld to decrease more than this in one time step
	restrat_threshold = pwp.dt / 3600 * 25

	if (prev_mld - mld) > restrat_threshold:
		mld_idx = np.argmin(np.abs(pwp.z - prev_mld))
		pwp._mix(mld_idx, n)
		mld = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold)
		return mld
	else:
		return mld

def mld_kara_mix(pwp, n):
	'''
	Calcultes MLD using Kara algorithm but mixes to prevent restratification.
	Model allows for MLD to rise by 25m per hour, if MLD decreases by more than this amount,
	the column is mixed from the previous MLD to the surface and MLD is recalculated
	:param pwp: instance of PWP model
	:param n: index of time step
	:return: mixed layer depth in meters [float]
	'''

	mld = mld_calc(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh)

	if n == 0:
		return mld

	# dens_threshold = 1e-4
	# mld2 = MLD_density_diff(pwp.sal[:, n], pwp.temp[:, n], pwp.z, dens_threshold)
	#
	# if mld2 < mld:
	#     neutral_strat = np.argmin(np.abs(pwp.z - mld2))
	#     pwp._mix(neutral_strat, n)
	#
	# mld = mld_calc(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh)
	# return mld

	prev_mld = pwp.mld[n-1]
	# restratification threshold: don't allow mld to decrease more than this in one time step
	restrat_threshold = pwp.dt / 3600 * 25

	if (prev_mld - mld) > restrat_threshold:
		mld_idx = np.argmin(np.abs(pwp.z - prev_mld))
		pwp._mix(mld_idx, n)
		mld = mld_calc(pwp.sal[:, n], pwp.temp[:, n], pwp.z, pwp.mld_thresh)
		return mld
	else:
		return mld

def momentum_flux_original(pwp, n, mld_idx):
	'''
	This is the original parametrization of momentum flux as writen
	by Earle Wilson
	:param pwp: instance of PWP object
	:param n: model time step index
	:param mld_idx: index of mixed layer depth
	:return:
	'''
	### Rotate u,v do wind input, rotate again, apply mixing ###
	ang = -pwp.f * pwp.dt / 2
	uvel, vvel = rotate(pwp.u_water[:,n], pwp.v_water[:,n], ang)
	du = (pwp.tx[n - 1] / (pwp.mld[n] * pwp.dens[0,n])) * pwp.dt
	dv = (pwp.ty[n - 1] / (pwp.mld[n] * pwp.dens[0,n])) * pwp.dt
	pwp.u_water[:mld_idx,n] = uvel[:mld_idx] + du
	pwp.v_water[:mld_idx,n] = vvel[:mld_idx] + dv

	### Apply drag to current ###
	# Original comment: this is a horrible parameterization of inertial-internal wave dispersion
	if pwp.drag_ON:
		ucon = 0.1 * np.abs(pwp.f)
		if ucon > 1e-10:
			pwp.u_water[:,n] *= (1 - pwp.dt * ucon)
			pwp.v_water[:,n] *= (1 - pwp.dt * ucon)
	else:
		print("Warning: Parameterization for inertial-internal wave dispersion is turned off.")
		printDragWarning = False

	pwp.u_water[:,n], pwp.v_water[:,n] = rotate(pwp.u_water[:,n], pwp.v_water[:,n], ang)

def rotate(x,y,theta):
	'''
	Rotates the vector (x,y) through an angle theta
	:param x: x component of vector
	:param y: y component of vector
	:param theta: angle [radians]
	:return: vector (x', y') as tuple
	'''
	r = (x + 1j * y) * np.exp(1j * theta)
	x = r.real
	y = r.imag

	return x, y