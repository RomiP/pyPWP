'''
This is an implementation of the Wanninkhof (2014) air-sea gas flux
originally by Rik Wanninkhof (Atlantic Oceanographic and Meteorological Laboratory of NOAA)
Relationship between wind speed and gas exchange over the ocean revisited
DOI 10.4319/lom.2014.12.351

Romina Piunno
Department of Physics
University of Toronto
29 June 2020
'''

'''
Note for future development:
to add new gasses, define a new function following the same template
as those already included and add the function name to the schmidt_num
dictionary
'''

import pyseaflux as psf
import seawater as sw

KELVIN = 273.15
mol = 6.02e23
R = 8.314 # J / K mol (gas constant)
g = 9.81 # m/s^2

# molar mass kg / mol
rho = {
	'o2': 0.032,
	'co2': 0.044,
	'air_dry': 0.02896
}

rho_air = 1.225 # kg/m^3


def schmidt_num_CO2(t, fresh_water=False):
	'''
	Schmidt number versus temperaturde for seawater (35‰) at temperatures from –2°C to 40°C
	Wanninkhof 2014
	Relationship between wind speed and gas exchange over the ocean revisited
	***FOR CO2***

	:param t: temperature (degrees C)
	:param fresh_water: boolean for fresh or sea water
	:return: schmidt number for CO2
	'''
	sc = 2116.8 - 136.25 * t + 4.7353 * t ** 2 - 0.092307 * t ** 3 + 0.0007555 * t ** 4

	if fresh_water:
		sc = 1923.6 - 125.06 * t + 4.3773 * t ** 2 - 0.085681 * t ** 3 + 0.00070284 * t ** 4

	return sc


def schmidt_num_O2(t, fresh_water=False):
	'''
	Schmidt number versus temperature for seawater (35‰) at temperatures from –2°C to 40°C
	Wanninkhof 2014
	Relationship between wind speed and gas exchange over the ocean revisited
	***FOR O2***

	:param t: temperature (degrees C)
	:param fresh_water: boolean for fresh or sea water
	:return: schmidt number for O2
	'''
	sc = 1920.4 - 135.6 * t + 5.2122 * t ** 2 - 0.10939 * t ** 3 + 0.00093777 * t ** 4

	if fresh_water:
		sc = 1745.1 - 124.34 * t + 4.8055 * t ** 2 - 0.10115 * t ** 3 + 0.00086842 * t ** 4

	return sc


# dictionary for appropriate schmidt num keyed by gas name
schmidt_num = {'co2': schmidt_num_CO2,
               'o2': schmidt_num_O2}


def transfer_velocity(u, v, t, gas):
	'''
	Computes air-sea transfer velocity of CO2
	:param u: wind component (m/s)
	:param v: wind component (m/s)
	:param t: Skin temp (degrees C)
	:param gas: string of gas name e.g. 'co2', 'o2'
	:return: transfer velocity (m/s)
	'''
	# calculate the appropriate schmidt number (based on gas)
	s_num = schmidt_num[gas](t)
	# compute in cm/h according to Wanninkhof paper
	k = 0.251 * (u ** 2 + v ** 2) * (660. / s_num) ** 0.5
	# convert cm/h to m/s
	k /= 3.6e5

	return k


def solubility(s, t, gas):
	'''
	Solubility of dissolved gasses in seawater at 1 atm
	:param t: Temperature (deg C)
	:param s: Salinity (PSU)
	:param gas: Name of dissolved gas (e.g. 'o2')
	:return: Solubility in kg / m^3 atm
	'''

	if gas == 'o2':
		k = sw.satO2(s, t)  # mL / L atm
		p = 1e5  # Pa
		t = 25 + 273.15  # K
		v = k/1000.

		n = p * v / (R * t)

	elif gas == 'co2':
		n = psf.solubility.solubility_weiss1974(s, t + KELVIN, press_atm=1)  # mol / L atm

	m = rho[gas] * n  # kg / L atm
	k = m * 1000  # kg / m^3 atm
	return  k

def partial_pressure_ocean(concentration, gas):
	'''
	Convert concentration (kg/m^3) in seawater to partial pressure (atm)
	:param concentration: kg gas / m^3 seawater
	:param gas: name of gas (e.g. 'o2')
	:return: partial pressure (atm)
	'''

	# Henry's Constant at STP in water (L atm / mol)
	H = {
		'o2': 770,
	}

	return concentration * H[gas]  / 1e3 / rho[gas]

def partial_pressure_atm(concentration, temp, gas):
	'''
	Convert concentration (kg/m^3) in atmosphere to partial pressure (pa)
	:param concentration: mass concentration of chemical in atmosphere
	:param temp: air temperature (deg C)
	:param gas: name of gas (e.g. 'o2')
	:return: partial pressure (pa)
	'''

	n = concentration / rho[gas]
	v = 1 # m^3
	t = temp + KELVIN

	p = n*R*t/v

	return p

def bubble_pressure(u, c_a, gas):
	'''
	Compute partial pressure of gas in bubble
	Stanley et al., 2009 https://doi.org/10.1029/2009JC005396
	Eqn 5
	:param u: 10 m wind speed (m/s)
	:param c_a: concentration of gas in atmosphere (kg/m^3)
	:param gas: string of gas name e.g. 'co2', 'o2'
	:return: partial pressure of <gas> in bubble (pa)
	'''

	z = 0.15*u - 0.55 # bubble depth (Eqn 6)
	xi = (c_a / rho[gas]) / (rho_air/ rho['air_dry']) # mol fraction
	p_atm = 101325 # pa
	rho_sw = 1027 # kg/m^3

	p_ib = xi * (p_atm + rho_sw*g*z)
	return p_ib

def diffusivegasflux(Ca, Cw, u, v, s, t, gas):
	'''
	computes air-sea diffusive gas flux (positive is water -> air)
	:param Ca: atmospheric boundary layer gas concentration (mass volume^-1)
	:param Cw: ocean boundary layer gas concentration (mass volume^-1)
	:param u: wind component (m/s)
	:param v: wind component (m/s)
	:param t: Skin temp (degrees C)
	:param gas: string of gas name e.g. 'co2', 'o2'
	:return: gas flux (mass area^-1 time^-1)
	'''

	gas = gas.lower()
	Ca = partial_pressure_atm(Ca, t, gas) / 101325 # convert partial pressure pa -> atm
	F = transfer_velocity(u, v, t, gas) * solubility(s, t, gas) * (partial_pressure_ocean(Cw, gas)- Ca)
	return F

def completely_trapped_bubbles(u, t, ca, gas):
	'''
	Compute mass flux driven by air injection of completely
	trapped bubbles (positive is into ocean)
	Stanley et al., 2009 https://doi.org/10.1029/2009JC005396
	Eqn 3
	:param u: 10m wind speed, magnitude (m/s)
	:param t: ocean temperature (deg C)
	:param p_ia: concentration of gas in atmosphere (kg/m^3)
	:gas: string of gas name e.g. 'co2', 'o2'
	:return: gas flux (mass area^-1 time^-1)
	'''

	p_a = partial_pressure_atm(ca, t, gas)
	ac = 9.1e-11 # s^2/m^2
	t += KELVIN
	f = ac * (u-2.27) ** 3 * p_a / (R * t) # mol / m^2 s

	f *= rho[gas] # kg / m^2 s
	return f
	# return max(0, f)

def partially_dissolved_bubbles(u, t, c_a, c_w, di, gas):
	'''
	Compute mass flux driven by air injection of completely
	trapped bubbles (positive is into ocean)
	Stanley et al., 2009 https://doi.org/10.1029/2009JC005396
	Eqn 4
	:param u: 10m wind speed, magnitude (m/s)
	:param t: ocean temperature (deg C)
	:param c_a: concentration of gas in atmosphere (kg/m^3)
	:param c_w: concentration of gas in ocean (kg/m^3)
	:param di: diffusivity coefficient (m^2/s)
	:gas: string of gas name e.g. 'co2', 'o2'
	:return: gas flux (mass area^-1 time^-1)
	'''

	p_iw = partial_pressure_ocean(c_w, gas) * 101325 # convert atm -> pa
	# p_ia = partial_pressure_atm(c_a, t, gas)
	p_ib = bubble_pressure(u, c_a, gas)

	t += KELVIN
	ap = 2.3e-3 # s^2/m^2
	alpha = 0.033 # colt 2012, 10.1016/B978-0-12-415916-7.00002-4
	d0 = 1. # normalisation factor (s/m^2)
	f = ap * (u - 2.7) ** 3 * alpha * (di/d0) ** (2/5) * (p_ib - p_iw) / (R * t)

	f *= rho[gas]  # kg / m^2 s

	return  f
	# return max(f, 0)

def saturation(s, t, gas):
	# STP conditions
	p = 1e5 # Pa
	R = 8.314 # J/(K mol)
	T = 273.15 # K
	if gas == 'o2':
		sat = sw.satO2(s, t) # mL / L = L / m^3
		v = sat / 1000 # m^3 O2 / m^3 water
		n = p*v/(R*T) # mol / m^3 water
		rho = 0.032 # kg / mol
		return  n*rho # kg / m^3




if __name__ == '__main__':
	import numpy as np
	t = np.linspace(-2, 10, 20)
	kelvin = 273.15
	print(transfer_velocity(5, 0, 4, 'o2'))
