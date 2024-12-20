'''
This is a Python implementation of the Price Weller Pinkel (PWP) 1D model.
Price, J. F., Weller, R. A., & Pinkel, R. (1986).
Diurnal cycling: Observations and models of the upper ocean response to
diurnal heating, cooling, and wind mixing.
Journal of Geophysical Research: Oceans, 91(C7), 8411–8427.
https://doi.org/10.1029/JC091iC07p08411

This code is based on the implementation of the PWP model originally
written by Earle Wilson (https://github.com/earlew/pwp_python_00)
(University of Washington, School of Oceanography, 18 Apr. 2016)
The major modification from the original is the object oriented
structure, allowing for simple interchangeability of surface flux
parametrizations.

Romina Piunno
Department of Physics
University of Toronto
31 Aug 2020
'''


from pwp.parametrizations import  *
from tqdm import tqdm
import os
import sys
import numpy as np
from scipy import io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seawater as sw
import xarray as xr

# specify surface flux parametrizations
fluxes = {
	'temp':temperature_flux,
	'sal':salinity_flux,
	'momentum':momentum_flux_original,
	'co2':wanninkoff,
	'o2':wanninkoff
}

# specify MLD parametrization
# mld_fn = mld_kara
# mld_fn = mld_kara_mod
mld_fn = mld_kara_mix
# mld_fn = mld_density_diff_search
# mld_fn = mld_density_diff_nearest
# mld_fn = mld_density_diff_adjusted

class PWP:
	def __init__(self):
		'''
		Initializes PWP model with default settings and paramters. All forcings
		are turned on.
		**************************************************************************************
		*** Changes to temporal or spatial step size must be done PRIOR to reading in data ***
		**************************************************************************************
		Fields initialized to "None" must be filled by calling read_in_init_data() prior
		to running the model
		'''

		# True/False flag to turn ON/OFF forcing.
		self.winds_ON = True # winds
		self.emp_ON = True # freshwater
		self.heat_ON = True # heat flux
		self.drag_ON = True # current drag due to internal-inertial wave breaking
		self.diffusion_ON = True # background molecular diffusion

		# prevent freezing via convetion where there is no ice formation
		self.allow_freezing = False

		# model settings
		self.dt = 3600. # time step [seconds]
		self.dz = 1.0 # depth increment [meters]
		self.max_depth = 1000. # Max depth of vertical coordinate [meters]
		self.mld_thresh = 0.2 # temperature change criterion for MLD [deg C]

		# physical constants
		self.rb = 0.65 # critical bulk richardson number [dimensionless]
		self.rg = 0.25  # critical gradient richardson number [dimensionless]
		self.rkz = 1.0e-5 # background vertical diffusion [m**2/s]
		self.lambda_red =  0.6 # longwave extinction coefficient [meters]
		self.lambda_blue = 20.  # shortwave extinction coefficient [meters]

		# profile metadata
		self.t_ref = '' # string denoting reference time
		self.lat = None # [deg N]
		self.lon = '' # [deg E]
		self.f = None # Coriolis frequency [s^-1]
		self.tracer_names = [] # string key words for tracer names

		# time and depth vectors
		self.time = None # seconds since t_ref
		self.z = None # meters below surface


		# parameters to keep track of for output
		self.temp = None # [deg C]
		self.sal = None # [psu]
		self.dens = None # [kg/m^3]
		self.mld = None # [m]
		self.u_water = None # [m/s]
		self.v_water = None # [m/s]

		# mandatory forcing to be read in
		# positive heat flux should mean ocean warming
		self.lw = None # longwave radiation [W/m^2]
		self.sw = None # shortwave radiation [W/m^2]
		self.qlat = None # latent heat flux [W/m^2]
		self.qsens = None # sensible heat flux [W/m^2]
		self.tx = None # zonal (E-W) wind shear [N/m^2]
		self.ty = None # meridional (N-S) wind shear [N/m^2]
		self.u = None # zonal (E-W) 10m wind speed [m/s]
		self.v = None # meridional (N-S) 10m wind speed [m/s]
		self.precip = None # Precipitation rate [m/s]

		# forcings to be calculated from fields read in
		self.evap = None # [m/s]
		self.emp = None # [m/s]
		self.q_in = None # [W/m^2]
		self.q_out = None # [W/m^2]

	def __str__(self):
		'''
		Returns a string representation of self, excluding vector parameters
		:return:
		'''
		s = ''
		params_forcing = ['winds_ON', 'emp_ON', 'heat_ON', 'drag_ON', 'diffusion_ON', 'allow_freezing']
		params_model_settings = ['dt', 'dz', 'max_depth', 'mld_thresh']
		params_constants = ['rb', 'rg', 'rkz', 'lambda_red', 'lambda_blue']
		params_profile_data = ['t_ref', 'lat', 'lon', 'tracer_names']

		s += '*** profile metadata ***\n'
		for i in params_profile_data:
			s += '{0}: {1}\n'.format(i, vars(self)[i])

		s += '\n*** physical constants ***\n'
		for i in params_constants:
			s += '{0}: {1}\n'.format(i, vars(self)[i])

		s += '\n*** model settings ***\n'
		for i in params_model_settings:
			s += '{0}: {1}\n'.format(i, vars(self)[i])

		s += '\n*** forcings ***\n'
		for i in params_forcing:
			s += '{0}: {1}\n'.format(i, vars(self)[i])


		return s

	def read_in_init_data(self, profile, met, tracer_list=[], time_step=None, depth_step=None):
		'''
		Reads in and stores initial profile and meteorology to self
		:param profile: path to profile file or Profile obj
			fields (* mandatory, - optional)
				* 'z' depth vector [m]
				* 's' salinity [psu]
				* 't' temperature [deg C]
				* 'lat' latitude [deg E]
				- 'lon' longitude [deg N]
				- 'u' zonal water profile speed [m/s]
				- 'v' meridional water profile speed [m/s]
				- tracers [kg/m^3]
		:param met: path to meteorology file or Meteorology obj
			fields (* mandatory, - optional)
				* 'time' time vector [days]
				* 'sw' shortwave radiation [W/m^2]
				* 'lw' longwave radiation [W/m^2]
				* 'qlat' latent heat flux [W/m^2]
				* 'qsens' sensible heat flux [W/m^2]
				* 'tx' zonal wind shear [N/m^2]
				* 'ty' meridional wind shear [N/m^2]
				* 'u' zonal wind speed [m/s]
				* 'v' meridional wind speed [m/s]
				* 'precip' precipitation rate [m/s]
				- tracers (atmospheric concentration) [kg/m^3]
				- 't_ref' reference time & date (string)
		:param tracer_list: list of tracer names to include e.g. ['co2', 'o2']
		:param time_step: (optional) opportunity to update default time step [hours]
		:param depth_step: (optional) opportunity to update default spatial step [meters]
		:return:
		'''

		if time_step:
			self.dt = time_step*3600.
		if depth_step:
			self.dz = depth_step

		# begin by loading in initial profile and forcing
		if isinstance(profile, str):
			prof_data = sio.loadmat(profile, squeeze_me=True)
		else:
			prof_data = profile.__dict__
		if isinstance(met, str):
			met_data = sio.loadmat(met, squeeze_me=True)
		else:
			met_data = met.__dict__

		# index of max depth in initial profile data
		prof_z = np.squeeze(prof_data['z'])
		# i_zmax = np.argmin(np.abs(prof_z - self.max_depth))
		# initialize depth vector based on step size and max depth
		self.z = np.arange(0, self.max_depth, self.dz)
		# initialize time vector based on step size and
		met_time = np.squeeze(met_data['time'])
		self.time = np.arange(0, met_time[-1] * 86400, self.dt)
		met_time *= 86400 # convert from days to sec

		# get number of time and depth increments
		nz = len(self.z)
		nt = len(self.time)

		# init output and forcing dicts
		self.mld = np.zeros(nt)
		self.dens = np.zeros([nz, nt])
		self.u_water = np.zeros([nz, nt])
		self.v_water = np.zeros([nz, nt])

		# read in latitude
		self.lat = np.mean(prof_data['lat'])

		# read in longitude if it's included
		if 'lon' in prof_data and prof_data['lon']:
			self.lon  = np.mean(prof_data['lon'])
		else:
			print('longitude not included in profile')

		# read in reference time if it's included
		try:
			self.t_ref = str(met_data['t_ref'][0])
		except:
			print('reference time not included in meteorology')

		# read in mandatory fields (interpolate to match new time and depth vectors)
		mandatory_fields = ['temp', 'sal', 'lw', 'sw', 'qlat', 'qsens', 'tx', 'ty', 'u', 'v', 'precip']

		for i in mandatory_fields:
			if i == 'temp':
				self.temp = np.zeros([nz, nt])
				intp_var = interp1d(prof_z, np.squeeze(prof_data['t']),
									axis=0, bounds_error=False, fill_value='extrapolate')
				self.temp[:, 0] = intp_var(self.z)
			elif i == 'sal':
				self.sal = np.zeros([nz, nt])
				intp_var = interp1d(prof_z, np.squeeze(prof_data['s']),
									axis=0, bounds_error=False, fill_value='extrapolate')
				self.sal[:, 0] = intp_var(self.z)
			else:
				intp_var = interp1d(met_time, np.squeeze(met_data[i]),
									axis=0, bounds_error=False, fill_value='extrapolate')
				vars(self)[i] = intp_var(self.time)


		# check if profile u and v are provided
		if 'u' in prof_data and 'v' in prof_data:
			if not (len(prof_data['u']) == 0 or len(prof_data['v']) == 0):
				intp_var = interp1d(prof_z, np.squeeze(prof_data['u']),
									axis=0, bounds_error=False, fill_value='extrapolate')
				self.u_water[:, 0] = intp_var(self.z)
				intp_var = interp1d(prof_z, np.squeeze(prof_data['v']),
									axis=0, bounds_error=False, fill_value='extrapolate')
				self.v_water[:, 0] = intp_var(self.z)

		# read in tracer init profile and forcing (interpolate to match new time and depth vectors)
		for i in tracer_list:
			try:
				tr_profile = np.squeeze(prof_data[i])
				tr_forcing = np.squeeze(met_data[i])
				intp_var = interp1d(prof_z, tr_profile,
									axis=0, bounds_error=False, fill_value='extrapolate')
				vars(self)[i] = np.zeros([nz, nt])
				vars(self)[i][:, 0] = intp_var(self.z)
				intp_var = interp1d(met_time, tr_forcing,
									axis=0, bounds_error=False, fill_value='extrapolate')
				vars(self)[i+'_forcing'] = intp_var(self.time)
				self.tracer_names.append(i)
			except:
				print('data for tracer {0} missing. Skipping...'.format(i))


		# now that all the data is read in, compute the remaining necessary fields
		self.f = sw.f(self.lat)
		self.dens[:,0] = sw.dens0(self.sal[:,0], self.temp[:,0])
		self.evap = -self.qlat/1000./2.5e6 # latent heat flux [W/m^2] / density [kg/m3] / latent heat of vap [J/kg]
		self.emp = self.evap - self.precip
		self.q_in = self.sw
		self.q_out = -(self.lw + self.qlat + self.qsens)
		self.mld[0] = mld_fn(self, 0)

	def plot_profile(self, var, n):
		'''
		Plots vertical profile of variable at a given time step and MLD (red line)
		:param var: name of variable e.g. 'temp'
		:param n: model time step index
		:return:
		'''

		x = vars(self)[var][:,n]
		plt.plot(x, self.z)
		line = np.ones(len(x))
		plt.plot(x, line * self.mld[n], color='r')
		plt.title('Time step: {0}, MLD: {1}m'.format(n, self.mld[n]))
		# invert y axis
		ax = plt.gca()
		ax.set_ylim(ax.get_ylim()[::-1])
		plt.show()

	def save(self, file_name, write_info=False):
		'''
		Saves instance of PWP as .mat or .nc file: dictionary structure
		:param file_name: string of where to save file e.g. path/to/file
		:return:
		'''

		path = os.path.dirname(file_name)
		name = os.path.basename(file_name)


		if not os.path.exists(path):
			os.mkdir(path)

		save_name = path + '/' + name
		if save_name.endswith('.nc'):
			dpwp = self.__dict__
			data = {}

			depth = dpwp.pop('z')
			time = dpwp.pop('time')
			coords = {
				'time':time,
				'depth':depth
			}
			for i in dpwp.keys():
				var = dpwp[i]
				if isinstance(var, np.ndarray):
					if min(var.shape) == 1 or len(var.shape) == 1:
						data[i] = (['time'], var)
					else:
						data[i] = (['depth', 'time'], var)
				else:
					coords[i] = var
			xr_pwp = xr.Dataset(
				data_vars=data,
				coords=coords
			)
			xr_pwp.to_netcdf(save_name)

		else:
			if not save_name.endswith('.mat'):
				save_name += '.mat'
			sio.savemat(save_name, self.__dict__)

		if write_info:
			save_name, ext = os.path.splitext(save_name)
			f = open(save_name + '_info.txt', 'w')
			f.write(self.__str__())
			f.close()

		print('saved successfully as', save_name)

	def run(self, save_name=None, write_info=False):
		'''
		Runs pwp model given initial conditions loaded in self. Must assure
		all required fields are provided
		:param save_name: file name and location of where to save output
		:param write_info: boolean whether or not to save model metadata as text file
		:return:
		'''

		# make sure user is aware results won't be saved if no filename is given
		if not save_name:
			while True:
				response = input('Are you sure you would like to run without saving? [y/n]')
				response = response.strip().lower()
				if response == 'y':
					print('running...')
					break
				elif response == 'n':
					print('terminating...')
					sys.exit()
				else:
					print('please enter \'y\' or \'n\'')

		tlen = len(self.time)
		# tlen = 5 # just for debugging


		for i in tqdm(range(1, tlen)):

			# carry profile from last time step
			self.sal[:, i] = self.sal[:, i - 1]
			self.temp[:, i] = self.temp[:, i - 1]

			# compute temp and sal changes
			if self.heat_ON:
				self.temp[:,i] += fluxes['temp'](self, i)
			if self.emp_ON:
				self.sal[0,i] = self.sal[0,i-1] + fluxes['sal'](self, i) # add surface flux

			# update the density
			self.dens[:,i] = sw.dens0(self.sal[:,i], self.temp[:,i])

			# apply tracer fluxes
			for j in self.tracer_names:
				# todo: check for saturation
				vars(self)[j][:,i] = fluxes[j](self, i, j)

			# # mix top few meters, this prevents freezing of the surface layer
			# # the seawater package has a bug whereby density decreases indefinitely
			# # past freezing point leading to an artificial stable equilibrium
			# # need to implement sub-time step mixing to prevent this
			# mix_depth = 2 * self.dt / 3600 # meters per hour
			# imix = np.argmin(abs(self.z - mix_depth))
			# self._mix(imix, i)

			# check freezing condition
			if self.allow_freezing:
				self._ice_formation(i)
			else:
				self._prevent_freezing(i)

			# free convection - remove static instability
			self._remove_si(i)

			# compute and record mixed layer depth
			self.mld[i] = mld_fn(self, i)
			mld_idx = np.argmin(np.abs(self.z-self.mld[i])) # index of mld

			# # for debugging
			# self.plot_profile('dens', i)

			# apply wind stress
			if self.winds_ON:
				fluxes['momentum'](self, i, mld_idx)

			# apply mixed layer entrainment
			self._bulk_richardson_mix(mld_idx, i)

			# apply shear flow instability
			self._grad_richardson_mix(i)

			# apply background diffusion
			if self.diffusion_ON:
				self._diffuse(i)


		print('Process complete.')
		if save_name:
			print('saving...')
			self.save(save_name, write_info)

	def _remove_si(self, n):
		'''
		Removes static instability - Simulates free convection
		unstable if there exists a step where density decreases with depth
		:param n: model time step
		:return:
		'''

		unstable = True

		while unstable:
			# compute density changes wrt depth
			d_dens = np.diff(self.dens[:,n])
			# use -0.01 as threshold to ignore numerical instability
			threshold = -1e-4
			if np.any(d_dens < threshold):
				# find 1st index where density decreases
				idx = np.min(np.argwhere(d_dens<threshold))
				# mix everything above static instability depth
				self._mix(idx+1, n)

			else:
				unstable = False

	def _bulk_richardson_mix(self, mld_idx, n):
		'''
		Removes mixed layer instability (mixed layer entrainment)
		via bulk richardson number relaxation
		:param mld_idx: depth index of MLD
		:param n: model time step index
		:return:
		'''

		# # alias density and speeds to keep it looking clean
		# dens = self.dens[:,n]
		# u = self.u_water[:,n]
		# v = self.v_water[:,n]

		for i in range(mld_idx, len(self.z)):
			h = self.z[i]
			# compute density and velocity discrete differentials
			d_dens = (self.dens[i, n] - self.dens[0, n]) / self.dens[0, n]
			d_vel2 = (self.u_water[i, n] - self.u_water[0, n]) ** 2 + \
					 (self.v_water[i, n] - self.v_water[0, n]) ** 2
			# compute the bulk richardson number see pwp eqn 9
			if d_vel2 == 0 or d_vel2 < 1e-10:
				Rb = np.inf
			else:
				Rb = g * h * d_dens / d_vel2

			# check stability criterion
			if Rb > self.rb:
				break # if stable
			else:
				self._mix(i, n) # apply mixing to above layers

	def _grad_richardson_mix(self, n):
		'''
		Removes shear flow instability via gradient richardson
		number relaxation
		:param n: model time step index
		:return:
		'''

		# # alias density and speeds to keep it looking clean
		# dens = self.dens[:, n]
		# u = self.u_water[:, n]
		# v = self.v_water[:, n]

		# initialize over whole profile, then narrow in
		j1 = 0
		j2 = len(self.z) - 1

		# mix to relax shear flow instability
		while True:

			j_range = np.arange(j1, j2)
			Rg = np.zeros(len(j_range))

			for j in j_range:
				# compute density and velocity discrete differentials
				d_dens = self.dens[j + 1, n] - self.dens[j, n]
				d_vel2 = (self.u_water[j + 1, n] - self.u_water[j, n]) ** 2 + \
						 (self.v_water[j + 1, n] - self.v_water[j, n]) ** 2
				# compute the bulk richardson number see pwp eqn 10
				if d_vel2 == 0 or d_vel2 < 1e-10:
					Rg[j - j_range[0]] = np.inf
				else:
					Rg[j - j_range[0]] = g / self.dens[j, n] * d_dens * self.dz / d_vel2

			# find the smallest value of Rb in the profile (where profile is most unstable)
			r_min = np.min(Rg)
			r_min_idx = np.argmin(Rg)

			# Check stability criterion
			if r_min > self.rg:
				break

			# Mix the cells r_min_idx and r_min_idx+1 that had the smallest Richardson Number
			self._stir(r_min, r_min_idx, n)

			# recompute the rich number over the part of the profile that has changed
			j1 = r_min_idx - 2
			if j1 < 1:
				j1 = 0

			j2 = r_min_idx + 2
			if j2 > len(self.z) - 1:
				j2 = len(self.z) - 1

	def _mix(self, idx, n):
		'''
		mixes everything above depth[idx] via computing the mean
		:param idx: index of mixing depth
		:param n: model time step
		:return:
		'''
		params = ['temp', 'sal', 'u_water', 'v_water'] + self.tracer_names
		for i in params:
			vars(self)[i][:idx+1, n] = np.mean(vars(self)[i][:idx+1, n])

		# compute resulting density profile
		self.dens[:, n] = sw.dens0(self.sal[:, n], self.temp[:, n])

	def _stir(self, rg, j, n):
		'''
		Partially mix two adjacent layers (j & j+1) to relax gradient
		richardson number
		:param rg: current grad. rich. num. at level j
		:param j: depth index of shear flow instability
		:param n: model time step index
		:return:
		'''

		rg_prime = self.rg + 0.05 # arbitrary, just need Rg' > Rg_crit for numerical convergence

		params = ['temp', 'sal', 'u_water', 'v_water'] + self.tracer_names
		for i in params:
			vars(self)[i][j, n] -= (1-rg/rg_prime)*(vars(self)[i][j, n] - vars(self)[i][j+1, n])/2.
			vars(self)[i][j+1, n] += (1 - rg / rg_prime) * (vars(self)[i][j, n] - vars(self)[i][j+1, n])/2.

		# compute resulting density profile
		self.dens[:, n] = sw.dens0(self.sal[:, n], self.temp[:, n])

	def _diffuse(self, n):
		'''
		Simulates background diffusion using finite differences of 1d heat equation
		:param n: model time step index
		:return:
		'''

		params = ['temp', 'sal', 'u_water', 'v_water'] + self.tracer_names
		F = self.rkz*self.dt/self.dz**2

		# check stability for numerical convergence
		if F > 0.5:
			print("WARNING: unstable CFL condition for diffusion! dt*rkz/dz**2 > 0.5.")
			print('Exiting model')
			sys.exit()

		for j in params:
			a = vars(self)[j][:,n-1]
			a_new = np.zeros(len(a))
			# boundary conditions
			a_new[0] = a[0]
			a_new[-1] = a[-1]

			# diffusion equation
			a_new[1:-1] = a[1:-1] + F*(a[:-2] - 2*a[1:-1] + a[2:])

			# update self
			vars(self)[j][:, n] = a_new

		# compute resulting density profile
		self.dens[:, n] = sw.dens0(self.sal[:, n], self.temp[:, n])

	def _ice_formation(self, n):
		# todo: model ice formation
		raise NotImplementedError

	def _prevent_freezing(self, n):
		'''
		Iteratively mix from the surface to consecutively deeper layers until the surface
		is above freezing temperature. This simulates sub-time step convection.
		:param n: time step index
		:return: None
		'''

		# freezing point of surface layer
		fp = sw.fp(self.sal[0,n], 0)

		mix_to = 1

		while self.temp[0,n] < fp:
			self._mix(mix_to, n)
			mix_to += 1
