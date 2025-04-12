
import pwp.inputs.profile as pro

import numpy as np
import seawater as sw
import xarray as xr

def read_prof_data():
	argo_file = '../../../examples/LabSea_sample_data/2008_10_24|10_30_58|58.1N|-52.9E|D4901446_029.nc'
	argoxr = xr.open_dataset(argo_file)
	latitude = float(argoxr['LATITUDE'])

	depth = sw.dpth(np.squeeze(argoxr['PRES_ADJUSTED']), latitude)
	temp =  np.squeeze(argoxr['TEMP_ADJUSTED'])
	sal = np.squeeze(argoxr['PSAL_ADJUSTED'])
	# prof = pro.Profile(
	# 	temp, sal,
	# 	depth,
	# 	latitude
	# )

	return temp, sal, depth, latitude

def test_make_profile_simple():
	'''
	Check Profile is properly initialised with the given data
	:return:
	'''

	t, s, z, lat = read_prof_data()

	prof = pro.Profile(t, s, z, lat)

	assert prof.lat == lat
	assert all(prof.t == t)
	assert all(prof.s == s)
	assert all(prof.z == z)
	assert all(prof.d == sw.dens0(s, t))

	assert all(i == len(prof.z) for i in [len(prof.s), len(prof.t), len(prof.d)])