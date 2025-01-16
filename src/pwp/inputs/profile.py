from pwp.check_si import smooth_si
from scipy import io as sio
import seawater as sw
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import datetime as dt
import pwp.MLD_calc as MLD_calc


class Profile:

	def __init__(self, temp, sal, depth, lat, lon='', tracers={}, u=[], v=[]):
		self.lat = lat
		self.lon = lon
		self.t = temp
		self.s = sal
		self.d = sw.dens0(self.s, self.t)
		self.z = depth
		self.u = u
		self.v = v
		self.tracers = list(tracers.keys())
		for i in tracers.keys():
			vars(self)[i] = tracers[i]

	def remove_si(self, start_ind=5):

		d_hat = sw.dens0(self.s, self.t)
		while np.any(np.diff(d_hat[start_ind:]) < 0):
			try:
				self.s, self.t = smooth_si(self.s, self.t, start_ind)
				d_hat = sw.dens0(self.s, self.t)
			except:
				print('Could not smooth profile')
				return

		# update density
		self.d = sw.dens0(self.s, self.t)

	def plot_profile(self, var, xlabel='', title='', savename=''):
		h = plt.plot(vars(self)[var], self.z)
		ax = plt.gca()
		ax.set_ylim(ax.get_ylim()[::-1])
		plt.ylabel('Depth (m)')
		plt.xlabel(xlabel)
		plt.title(title)
		if savename:
			plt.savefig(savename)
		plt.show()
		return ax

	def save(self, path, filename):
		if not os.path.exists(path):
			os.mkdir(path)

		test = vars(self)
		sio.savemat(path + '/' + filename, vars(self))


# def read_from_armor(armor_file, lat, lon):
# 	dat = Dataset(armor_file)
#
# 	if lon < 0:
# 		lon = 360 + lon
#
# 	lat_vec = np.squeeze(dat['latitude'][:])
# 	lon_vec = np.squeeze(dat['longitude'][:])
# 	time = np.squeeze(dat['time'][:])
# 	z = np.squeeze(dat['zo'][:])
#
# 	i_lat = np.argmin(np.abs(lat_vec - lat))
# 	i_lon = np.argmin(np.abs(lon_vec - lon))
#
# 	# dim = [time, depth, lat, lon]
# 	# temp = np.array(dat['to'][:])
# 	temp = np.array(dat['to'][:, :, i_lat, i_lon])
#
# 	print()
# 	return
#
#
# def read_from_argo(argo_file):
# 	dat = Dataset(argo_file)
#
# 	lat = float(dat['LATITUDE'][:])
# 	lon = float(dat['LONGITUDE'][:])
#
# 	pres = np.squeeze(dat['PRES'])  # dbar
# 	sal = np.squeeze(dat['PSAL'])  # psu
# 	temp = np.squeeze(dat['TEMP'])  # deg C
#
# 	dat.close()
#
# 	z = sw.dpth(pres, lat)
#
# 	prof = Profile(temp=temp, sal=sal, depth=z, lat=lat, lon=lon)
#
# 	return prof


def build_from_existing(profile_path, savename=''):
	# profile_dat = sio.loadmat('/Volumes/Lovelace/PWP_data/test_data/Greenland_Iceland_test_data/ArgoFall_IS_PWP.mat')
	#
	# test_profile = sio.loadmat('/Users/rominapiunno/PycharmProjects/PWP_romi/input_files/test_profile_LabSea.mat')
	profile_dat = sio.loadmat(profile_path, squeeze_me=True)

	# lat = 66.9
	# lon = -12.6
	lat = float(profile_dat['lat'])
	lon = float(profile_dat['lon'])
	s = np.squeeze(profile_dat['s'])
	t = np.squeeze(profile_dat['t'])
	z = np.squeeze(profile_dat['z'])
	tracers = profile_dat['tracers']
	u = profile_dat['u']
	v = profile_dat['v']

	prof = Profile(t, s, z, lat, lon, tracers=tracers, u=u, v=v)

	# start_ind = 5
	# d_hat = sw.dens0(prof.s, prof.t)
	# while np.any(np.diff(d_hat[start_ind:]) < 0):
	#     sal, temp = smooth_si(prof.s, prof.t, start_ind)
	#     d_hat = sw.dens0(sal, temp)
	#     diff = np.diff(d_hat[start_ind:])
	#     prof.s = sal
	#     prof.t = temp
	#     prof.d = d_hat

	# plt.plot(prof.d, z)
	# # plt.plot(d_hat, z)
	# ax = plt.gca()
	# ax.set_ylim(ax.get_ylim()[::-1])
	# plt.show()

	if savename:
		# sio.savemat('/Users/rominapiunno/PycharmProjects/PWP_romi/input_files/test_profile_IcelandSea.mat', prof.__dict__)
		sio.savemat(savename, prof.__dict__)

	return prof


def read_from_problematic_BGC():
	file_path = '/Volumes/KatherineJ/LabSea_DeepConvetion_Profiles/argo/BGC/argo-profiles-3901668.nc'
	dat = Dataset(file_path)

	t_ref = dt.datetime(1950, 1, 1, 0, 0, 0)

	lat = np.squeeze(dat['LATITUDE'][:])
	lon = np.squeeze(dat['LONGITUDE'][:])

	pres = np.array(dat['PRES_ADJUSTED'][:])
	temp = np.array(dat['TEMP_ADJUSTED'][:])
	sal = np.array(dat['PSAL_ADJUSTED'][:])
	oxy = np.array(dat['DOXY_ADJUSTED'][:])
	date = np.array(dat['JULD'][:])

	# subset the good values
	dTime = dt.timedelta(float(date[0]))
	lat = lat[0]
	lon = lon[0]
	p = pres[0, :]
	t = temp[0, :]
	s = sal[0, :]
	o2 = oxy[2, :]
	z = sw.dpth(p, lat)

	# convert from micromole/kg -> kg/m^3
	ibad = np.argmin(np.abs(o2 - 99999))  # filter out where data is missing
	o2[ibad:] = o2[ibad - 1]
	rho = sw.dens0(s, t)
	o2 = o2 * 3.2e-8 * rho

	day = t_ref + dTime
	strDate = day.strftime("%Y_%m_%d|%H_%M_%S")

	prof = Profile(temp=t, sal=s, depth=z, lat=lat, lon=lon)
	prof.o2 = o2
	mld = mld_calc(s, t, z, 0.2)
	imld = np.argmin(np.abs(z - mld))
	# prof.remove_si(start_ind=imld)

	file_name_mod = strDate + '|{0}N{1}W|'.format(abs(int(lat)), abs(int(lon)))
	return prof, file_name_mod

def N2(s, t, z):
	'''
	N^2 = -g/rho_0 * drho / dz
	:param s:
	:param t:
	:param z:
	:return:
	'''

	g = 9.81  # m/s^2
	rho_0 = 1000.  # kg/m^3
	rho = sw.dens0(s,t)
	N = -g/rho_0*(np.diff(rho)/np.diff(z))
	return N


if __name__ == '__main__':
	# file_path = '/Volumes/Lovelace/PWP_data/test_data/LabSea-armor-3d-nrt-weekly_1595285840003.nc'
	# read_from_armor(file_path, lat=55, lon=-55)

	# file_path = '/Volumes/Lovelace/PWP_data/test_data/argo-58N54W-R4902469_021.nc'
	# prof = read_from_argo(file_path)

	# prof, fname = read_from_problematic_BGC()

	# # prof.plot_profile('d')
	# prof.remove_si(start_ind=25)
	# prof.plot_profile('d', 'Density ($kg/m^3$)')
	# sio.savemat('/Volumes/Lovelace/PWP_data/profiles/LabSea-{0}.mat'.format(fname),
	#             prof.__dict__)

	# %% plot single profile

	# path = '/Volumes/KatherineJ/PWP_obs_profiles/LabSea_DeepConvection/argo_all/' \
	#        '2019_10_29|05_20_00|57N|-53E.mat'
	# # path = '/Volumes/KatherineJ/PWP_obs_profiles/LabSea_DeepConvection/argo_all/' \
	# #        '2020_04_01|05_21_00|56N|-53E.mat'
	# saveroot = '/Users/rominapiunno/Desktop/'
	# prof = build_from_existing(path)
	#
	# # prof.plot_profile('t', 'Temperature ($^{\circ} C$)', savename=saveroot+'temp.png')
	# # prof.plot_profile('d', 'Density ($Kg/m^3$)', savename=saveroot+'dens.png')
	# drho = np.diff(prof.d)
	# plt.plot(prof.z[:-1], drho)
	# plt.xlabel('Depth (m)')
	# plt.ylabel('Density Gradient ($Kg/m^4$)')
	# plt.savefig('/Users/rominapiunno/Desktop/drhodz.png')
	# plt.show()

	# %% make climo profile with extremes
	root = '/Volumes/Marvin/PWP_data/LabSea-profiles/climo_profiles/'
	climofile = 'LabSea_deep_conv_profs_2002-2023_10_argo.mat'

	data = sio.loadmat(root+climofile, squeeze_me=True)

	n = data['dens'].shape[1]
	n2_min = np.inf
	imin = 0
	n2_max = -np.inf
	imax = 0
	# for i in range(n):
	# 	s = data['sal'][:, i]
	# 	t = data['temp'][:,i]
	# 	x = smooth_si(s, t, i_start=50)
	# 	if not x is None:
	# 		s,t = x
	# 		n2 = N2(s, t, data['z'])
	# 		n2 = np.mean(n2)
	# 		if n2 < n2_min:
	# 			n2_min = n2
	# 			imin = i
	# 		if n2 > n2_max and i!=10:
	# 			n2_max = n2
	# 			imax = i
	#
	# print(imax, n2_max)
	# print(imin, n2_min)
	#
	lat = 57.4
	# p_min = Profile(data['temp'][:,imin], data['sal'][:,imin], data['z'], lat)
	# p_min.remove_si(50)
	# p_min.plot_profile('d')
	# p_min.save(root,'LabSea_deep_conv_minN2_2002-2023_10.mat')
	#
	# p_max = Profile(data['temp'][:,imax], data['sal'][:,imax], data['z'], lat)
	# p_max.remove_si(50)
	# p_max.plot_profile('d')
	# p_min.save(root, 'LabSea_deep_conv_maxN2_2002-2023_10.mat')

	p_mean = Profile(data['m_temp'], data['m_sal'], data['z'], lat)
	p_mean.remove_si(50)
	p_mean.plot_profile('d')
	p_mean.save(root, 'LabSea_deep_conv_climo_2002-2023_10.mat')



