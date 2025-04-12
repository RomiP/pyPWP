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

# molar mass kg / mol
rho = {
	'o2': 0.032,
	'co2': 0.044,
}


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

def partial_pressure(concentration, gas):
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



def gasflux(Ca, Cw, u, v, s, t, gas):
	'''
	computes air-sea gas flux (positive is water -> air)
	:param Ca: atmospheric boundary layer gas concentration (mass volume^-1)
	:param Cw: ocean boundary layer gas concentration (mass volume^-1)
	:param u: wind component (m/s)
	:param v: wind component (m/s)
	:param t: Skin temp (degrees C)
	:param gas: string of gas name e.g. 'co2', 'o2'
	:return: gas flux (mass area^-1 time^-1)
	'''

	gas = gas.lower()
	F = transfer_velocity(u, v, t, gas) * solubility(s, t, gas) * (partial_pressure(Cw, gas)- Ca)
	return F


if __name__ == '__main__':
	import numpy as np
	t = np.linspace(-2, 10, 20)
	kelvin = 273.15
	print(transfer_velocity(5, 0, 4, 'o2'))
