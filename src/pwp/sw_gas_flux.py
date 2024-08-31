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


def schmidt_num_CO2(t, fresh_water=False):
	'''
	Schmidt number versus temperature for seawater (35‰) at temperatures from –2°C to 40°C
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
	k = 0.251 * (u ** 2 + v ** 2) * (660. / s_num) ** -0.5
	# convert to m/s
	k *= 0.1 / 3600.

	return k


def gasflux(Ca, Cw, u, v, t, gas):
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

	F = transfer_velocity(u, v, t, gas.lower()) * (Cw - Ca)
	return F
