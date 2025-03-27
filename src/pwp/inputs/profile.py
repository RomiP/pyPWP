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
		'''
		Construct ocean profile
		:param temp: temperature (deg C)
		:param sal: salinity (psu)
		:param depth: increasing from ocean surface (m)
		:param lat: latitude (deg N)
		:param lon: longitude (deg W)
		:param tracers: dict tracer_name:[measurements]
		:param u: zonal velocity (m/s)
		:param v: meridional velocity (m/s)
		'''

		for i in [temp, sal] + list(tracers.values()):
			assert len(i) == len(depth)
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


	def __eq__(self, other):
		for i in vars(self):
			if self[i] != other[i]:
				return False
		return True

	def remove_si(self, min_depth=5):
		'''
		Remove static instability from profile by iteratively averaging sal and
		temp until density is non-decreasing with depth
		This does not mix other parameters such as tracers
		:param min_depth: depth in m after which to start checking for instability
		(effectively skip over the boundary layer)
		:return: None
		'''

		start_ind = np.argmin(np.abs(self.z - min_depth))
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
		'''
		Create a vertical line plot of <var> with depth
		:param var: variable name [s, t, d, z, u, v, <tracer name>]
		:param xlabel: horizontal axis label
		:param title: Title axes label
		:param savename: filepath to save figure
		:return: axes
		'''
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

	def save(self, saveas):
		'''
		Saves profile as mat file
		:param saveas: filepath to destination
		:return: None
		'''
		path = os.path.dirname(saveas)
		# name = os.path.basename(saveas)
		if not os.path.exists(path):
			os.mkdir(path)

		# test = vars(self)
		sio.savemat(saveas, vars(self))


def build_from_existing(profile_path, savename=''):
	'''
	Open and init a new Profile with data saved from
	previously saved Profile instance.
	:param profile_path: path to file
	:param savename: re-save the data as mat file
	:return: Profile
	'''
	profile_dat = sio.loadmat(profile_path, squeeze_me=True)

	lat = float(profile_dat['lat'])
	lon = float(profile_dat['lon'])
	s = np.squeeze(profile_dat['s'])
	t = np.squeeze(profile_dat['t'])
	z = np.squeeze(profile_dat['z'])
	tracers = profile_dat['tracers']
	u = profile_dat['u']
	v = profile_dat['v']

	prof = Profile(t, s, z, lat, lon, tracers=tracers, u=u, v=v)


	if savename:
		sio.savemat(savename, prof.__dict__)

	return prof


