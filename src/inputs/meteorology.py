import netCDF4 as nc
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from scipy import io as sio
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset
import pickle
import re

def datenum2datetime(dt_num, refdate=dt.datetime(1,1,1)):
    '''
    Converts serial date (matlab convention by default) to datetime obj
    :param dt_num: matlab serial date (days since 0000-Jan-01 00:00:00)
    :param refdate: datetime obj
    :return: datetime object
    '''

    delta = dt.timedelta(days=dt_num - 367)
    return refdate + delta

def datetime2datenum(date, refdate=dt.datetime(1, 1, 1)):
    '''
    Converts datetime obj to serial date (matlab convention by default)
    :param date: datetime obj
    :param refdate: datetime obj
    :return: matlab serial date (days since 0000-Jan-01 00:00:00)
    '''

    datediff = date - refdate + dt.timedelta(days=367)
    return datediff.days + datediff.seconds/86400.

class Meteorology:

    def __init__(self, start_date, end_date, time_step, metdata=None, tracers=None):
        '''
        Initialize meteorological forcing for PWP between two times
        with given time step
        :param start_date: datetime obj
        :param end_date: datetime obj
        :param time_step: number in hours
        :param metdata: dict-like containing meteorology time series (key names must match attribute names)
        :param tracers: time series of tracer concentrations (atmo boundary layer), must contain 'time' as key
        '''
        # date & time
        self.t_ref = start_date.strftime('%Y-%m-%d %H:%M:%S') # yyyy-mm-dd HH:MM:SS
        N = (end_date-start_date).days
        self.time = np.linspace(0, N, N*round(24/time_step))  # days since t_ref
        # radiation W/m^2
        self.lw = np.zeros(len(self.time))
        self.sw = np.zeros(len(self.time))
        # heat fluxes W/m^2
        self.qlat = np.zeros(len(self.time))
        self.qsens = np.zeros(len(self.time))
        # wind stress N/m^2
        self.tx = np.zeros(len(self.time))
        self.ty = np.zeros(len(self.time))
        # 10m wind speed m/s
        self.u = np.zeros(len(self.time))
        self.v = np.zeros(len(self.time))
        # precipitation m/s
        self.precip = np.zeros(len(self.time))
        # passive tracer kg/m^3
        if tracers:
            self.tracers = list(tracers.keys() - 'time')
            for i in self.tracers:
                vars(self)[i] = self.interp_to_timestep(tracers['time'], tracers[i])
        # self.co2 = np.zeros(len(self.time))

        if metdata:
            t = metdata['time']
            if 'lw' in metdata:
                self.lw = self.interp_to_timestep(t, metdata['lw'])
            if 'sw' in metdata:
                self.sw = self.interp_to_timestep(t, metdata['sw'])
            if 'qlat' in metdata:
                self.qlat = self.interp_to_timestep(t, metdata['qlat'])
            if 'qsens' in metdata:
                self.qsens = self.interp_to_timestep(t, metdata['qsens'])
            if 'tx' in metdata:
                self.tx = self.interp_to_timestep(t, metdata['tx'])
            if 'ty' in metdata:
                self.ty = self.interp_to_timestep(t, metdata['ty'])

    def interp_to_timestep(self, t, data):
        '''
        given a timeseries, interpolate data to the time intervals of the model
        :param t: time step [hours since model ref time]
        :param data: values
        :return: Time series sampled at model time steps
        '''

        return np.interp(self.time, t, data)



    def read_from_file(self, filepath):
        met_file = sio.loadmat(filepath)
        met_time = np.squeeze(met_file['time'])

        # check that time vectors of input file match that of self
        if not len(self.time) == len(met_time):
            print('Time vectors do not match.')
            while True:
                response = input('Would you like to update the time vector to match the given file? y/n')
                if response.lower() == 'y':
                    tref = met_time[0]
                    if tref != 0:
                        met_time -= tref
                        self.t_ref = datenum2datetime(tref).strftime('%Y-%m-%d %H:%M:%S')
                        print('New reference date:', self.t_ref)
                    self.time = met_time[:]
                    print('Time vector has been updated')
                    break
                elif response.lower() == 'n':
                    print('Aborting process')
                    return


        for i in vars(self).keys():
            if i == 'time' or i == 't_ref':
                continue
            try:
                vars(self)[i] = np.squeeze(met_file[i])
            except:
                print('attribute {0} was not in file. Skipping...'.format(i))
                continue

    def set_radiation(self, lw=0, sw=0):
        '''
        Radiative forcing (W/m^2). Positive direction is
        from air to sea
        :param lw: long wave radiative forcing
        :param sw: short wave radiative forcing
        :return:
        '''
        if not np.isscalar(lw):
            if len(lw) != len(self.lw):
                print('Length of lw time series does not match. Aborting process')
                return
        if not np.isscalar(sw):
            if len(sw) != len(self.sw):
                print('Length of sw time series does not match. Aborting process')
                return


        self.lw = np.squeeze(lw)
        self.sw = np.squeeze(sw)

    def set_heat_flux(self, lat=0, sens=0):
        '''
        Heat flux (W/m^2). Positive direction is
        from air to sea
        :param lat: latent heat flux
        :param sens: sensible heat flux
        :return:
        '''
        if not np.isscalar(lat):
            if len(lat) != len(self.qlat):
                print('Length of latent heat time series does not match. Aborting process')
                return
        if not np.isscalar(sens):
            if len(sens) != len(self.qsens):
                print('Length of sensible heat time series does not match. Aborting process')
                return


        self.qlat = np.squeeze(lat)
        self.qsens = np.squeeze(sens)

    def set_wind_stress(self, tx=0, ty=0):
        '''
        Wind stress (N/m^2). Positive directions are towards
        North and East
        :param tx: zonal (W-E) wind stress
        :param ty: meridional (N-S) wind stress
        :return:
        '''
        if not np.isscalar(tx):
            if len(tx) != len(self.tx):
                print('Length of x-wind stress time series does not match. Aborting process')
                return
        if not np.isscalar(ty):
            if len(ty) != len(self.ty):
                print('Length of y-wind stress time series does not match. Aborting process')
                return


        self.tx = np.squeeze(tx)
        self.ty = np.squeeze(ty)

    def set_wind_speed(self, u=0, v=0):
        '''
        Wind speed (m/s). Positive directions are towards
        North and East
        :param u: zonal (W-E) wind speed
        :param v: meridional (N-S) wind speed
        :return:
        '''
        if not np.isscalar(u):
            if len(u) != len(self.u):
                print('Length of x-wind speed time series does not match. Aborting process')
                return
        if not np.isscalar(v):
            if len(v) != len(self.v):
                print('Length of y-wind speed time series does not match. Aborting process')
                return


        self.u = np.squeeze(u)
        self.v = np.squeeze(v)

    def set_precipitation(self, p):
        '''
        Fresh water forcing. Precipitation in mm/day
        All values should be non-negative (P>=0)
        :param p: precipitation rate
        :return:
        '''
        if not np.isscalar(p):
            if len(p) != len(self.precip):
                print('Length of precipitation time series does not match. Aborting process')
                return

        # convert from mm/day to m/s
        p = p/1000./86400.
        self.precip = np.squeeze(p)

    def plot_heat_winds(self):
        q_net = self.sw + self.lw + self.qlat + self.qsens
        winds = np.sqrt(self.u**2 + self.v**2)

        fig, ax1 = plt.subplots()

        col = 'tab:red'
        ax1.set_xlabel('Time (days since start date)')
        ax1.set_ylabel('Net heat flux into ocean ($W/m^2$)', color=col)
        ax1.plot(self.time, q_net, color=col)
        ax1.tick_params(axis='y', labelcolor=col)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        col = 'tab:blue'
        ax2.set_ylabel('Wind speed (m/s)', color=col)  # we already handled the x-label with ax1
        ax2.plot(self.time, winds, color=col)
        ax2.tick_params(axis='y', labelcolor=col)

        plt.title('Start Date {0}'.format(self.t_ref))
        plt.show()

    def save(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)
        sio.savemat(path+'/'+filename, vars(self))




