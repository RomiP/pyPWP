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

def add_const_co2_to_nc(filepath):
    # todo: finish writing this code
    my_file = nc.Dataset(filepath, 'r+')


    # time = my_file.variables['time']
    dim_size = my_file.dimensions['t'].size

    mm_co2 = 0.044  # kg/mol
    mm_air = 0.029  # kg/mol
    ppm_co2 = 400  # ppm
    rho_air = 1.2  # kg/m^3
    conc_co2 = rho_air / mm_air * ppm_co2 / 1.0e6 * mm_co2

    print('CO2 concentration {0}ppb = {1} kg/m^3'.format(ppm_co2, conc_co2))

    co2 = np.ones(dim_size) * conc_co2
    print(co2.shape)

    my_file.createVariable('co2', float, 't')
    my_file['co2'][:] = co2

    my_file.close()

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

    def __init__(self, start_date, end_date, time_step):
        '''
        Initialize meteorological forcing for PWP between two times
        with given time step
        :param start_date: datetime obj
        :param end_date: datetime obj
        :param time_step: number in hours
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
        self.co2 = np.zeros(len(self.time))

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

def make_test_met():
    root_path = '/Users/rominapiunno/PycharmProjects/pwp_python_00/input_data/'

    t1 = dt.datetime(1996, 12, 1, 3, 0, 0)
    dtnum1 = 729360.125
    t2 = dt.datetime(1997, 3, 31, 15, 0, 0)
    dtnum2 = 729480.625
    datetest = datenum2datetime(dtnum1)
    # datetest = datetime2datenum(t2)
    print(datetest)

    metfile = '/Users/rominapiunno/PycharmProjects/pwp_python_00/input_data/met.mat'
    met = Meteorology(t1, t2, 12)
    met.read_from_file(metfile)
    print(vars(met).keys())

    # add some phony wind data

    u = 10. * np.sin(met.time / met.time[-1])
    v = 5. * np.sin(met.time / met.time[-1] + np.pi / 4.)
    met.set_wind_speed(u, v)

    met.save('/Users/rominapiunno/Documents/PWP_data/input_data/', 'test.mat')

def extract_ERA5_ts(dir, varname, yrs, mnths, lat, lon):

    if lon < 0:
        lon = 360 + lon

    ts = np.array([])
    for y in yrs:
        for m in mnths:
            filename = '{0}_1hr/era5_{0}_{1}_{2}.nc'.format(varname, y, str(m).zfill(2))
            print(filename)
            dat = Dataset(dir+'/'+filename)
            lat_vec = np.squeeze(dat['latitude'][:])
            lon_vec = np.squeeze(dat['longitude'][:])
            i_lat = np.argmin(np.abs(lat_vec - lat))
            i_lon = np.argmin(np.abs(lon_vec - lon))
            var = np.squeeze(dat[varname][:])[:,i_lat, i_lon]
            ts = np.concatenate((ts, var))

    return ts

def ERA5_met():

    yr = 2020
    months = [1,2,3]
    lat = 57.9
    lon = -54.5

    d_start = dt.datetime(yr, months[0], 1)
    d_end = dt.datetime(yr, months[-1], 31)
    time_step = 1. # hours

    # met = Meteorology(d_start, d_end, time_step)
    infile = open('working_files/met.pkl', 'rb')
    met = pickle.load(infile, encoding='bytes')

    root = '/Volumes/Lovelace/ECMWF/ERA5/'

    # # read in lw and sw time series
    # sw = extract_ERA5_ts(root,'ssr',[2020], [1,2,3], lat=lat, lon=lon)
    # lw = extract_ERA5_ts(root,'str',[2020], [1,2,3], lat=lat, lon=lon)
    # # convert from J/m^2 to W/m^2 by dividing by time step in seconds
    # sw = sw/(time_step*3600.)
    # lw = lw/(time_step*3600.)
    # # set in meteorology
    # met.set_radiation(lw, sw)

    # # read in qlat and qsens time series
    # qlat = extract_ERA5_ts(root,'slhf',[2020], [1,2,3], lat=lat, lon=lon)
    # qsens = extract_ERA5_ts(root,'sshf',[2020], [1,2,3], lat=lat, lon=lon)
    # # convert from J/m^2 to W/m^2 by dividing by time step in seconds
    # qlat = qlat/(time_step*3600.)
    # qsens = qsens/(time_step*3600.)
    # # set in meteorology
    # met.set_heat_flux(qlat, qsens)

    # # read in u and v winds time series
    # u = extract_ERA5_ts(root,'u10',[2020], [1,2,3], lat=lat, lon=lon)
    # v = extract_ERA5_ts(root,'v10',[2020], [1,2,3], lat=lat, lon=lon)
    # # set in meteorology
    # met.set_wind_speed(u,v)

    # read in precip time series
    root = '/Volumes/KatherineJ/ECMWF/ERA5-NW/'
    precip = extract_ERA5_ts(root,'mtpr',[2020], [1,2,3], lat=lat, lon=lon)
    # # convert from m to m/s by dividing by time step in seconds
    # precip = precip/(time_step*3600.)
    precip = precip * 86400. # convert mm/s -> mm/day
    # set in meteorology
    met.set_precipitation(precip)

    # # read in surface stress time series
    # tx = extract_ERA5_ts(root,'ewss',[2020], [1,2,3], lat=lat, lon=lon)
    # ty = extract_ERA5_ts(root, 'nwss', [2020], [1, 2, 3], lat=lat, lon=lon)
    # # set in meteorology
    # met.set_wind_stress(tx, ty)

    # # read in air density and drag coef time series to calculate stress
    # rho = extract_ERA5_ts(root,'rhoao',[2020], [1,2,3], lat=lat, lon=lon)
    # cd = extract_ERA5_ts(root,'cdww',[2020], [1,2,3], lat=lat, lon=lon)
    # # calculate surface stress
    # tx = rho*cd*met.u
    # ty = rho*cd*met.v
    # # set in meteorology
    # met.set_wind_stress(tx, ty)

    met.save('/Volumes/Lovelace/PWP_data/meteorologies', 'LabSea_met-20200101-58N54W.mat')

    with open('working_files/met.pkl', 'wb') as output:
        pickle.dump(met, output, pickle.HIGHEST_PROTOCOL)

    print('it worked!')

def concat_data(root, gen_fname, vals):

    concat_dat = {}

    dt = None

    for i in vals:
        f = root + gen_fname.format(i)
        dat = sio.loadmat(f)
        keys = dat.keys()
        for k in keys:
            if k.startswith('_'):
                continue

            ts = np.squeeze(dat[k])
            if k == 'time':
                dt = np.mean(np.diff(ts))
                continue

            if k in concat_dat:
                concat_dat[k] = np.concatenate([concat_dat[k], ts])
            else:
                concat_dat[k] = ts


    key = list(concat_dat.keys())
    n = len(concat_dat[key[0]])
    concat_dat['time'] = np.arange(n)*dt

    return concat_dat

def climatological_met(sigma, extrm_var):

    dat_root = '/Volumes/KatherineJ/ERA5_LabSea_dat/DeepConvZone_ts/mean&std/'
    months = ['01', '02', '03']

    # # 1: mu+stt , 0: mu , -1 mu-std
    # sigma = 0.0

    t_start = dt.datetime(2019, 1, 1, 0, 0, 0)
    t_end = dt.datetime(2019, 4, 1, 0, 0, 0)
    met = Meteorology(t_start, t_end, 1.0)

    # radiation
    var = 'msnswrf'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    sw = mu
    if extrm_var == 'sw' or extrm_var == 'all':
        sw += sigma * std

    var = 'msnlwrf'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    lw = mu
    if extrm_var == 'lw' or extrm_var == 'all':
        lw += sigma * std

    met.set_radiation(lw, sw)

    # heat
    var = 'mslhf'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    qlat = mu
    if extrm_var == 'lhf' or extrm_var == 'all':
        qlat += sigma * std

    var = 'msshf'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    qsens = mu
    if extrm_var == 'shf' or extrm_var == 'all':
        qsens =+ sigma * std

    met.set_heat_flux(qlat, qsens)

    # wind stress
    var = 'ewss'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    tx = mu
    if extrm_var == 'tau' or extrm_var == 'all':
        tx += sigma * std
    # surface stress has units N m**-2 s
    tx = tx / 3600 # convert to N/m^2

    var = 'nsss'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    ty = mu
    if extrm_var == 'tau' or extrm_var == 'all':
        ty += sigma * std
    # surface stress has units N m**-2 s
    ty = ty / 3600  # convert to N/m^2

    met.set_wind_stress(tx, ty)

    # wind speed
    var = 'u10'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    u = mu
    if extrm_var == 'wind' or extrm_var == 'all':
        u += sigma * std

    var = 'v10'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    v = mu
    if extrm_var == 'wind' or extrm_var == 'all':
        v += sigma * std

    met.set_wind_speed(u, v)

    # precip
    var = 'mtpr'
    general_fname = 'era5_mean_std_' + var + '_{0}.mat'
    all_dat = concat_data(dat_root, general_fname, months)
    mu = np.squeeze(all_dat['mean'])
    std = np.squeeze(all_dat['std'])
    p = mu
    if extrm_var == 'precip' or extrm_var == 'all':
        p += sigma * std
    '''
     mtpr:
     1 kg of water spread over 1 square metre of surface is 
     1 mm deep (neglecting the effects of temperature on the 
     density of water), therefore the units are equivalent 
     to mm (of liquid water) per second
    '''
    p = p*86400. # convert to mm/day

    met.set_precipitation(p)

    # save
    save_name = 'LabSea_JFM_climo.mat'
    if sigma == 1:
        save_name = 'LabSea_JFM_climo_upper_{0}.mat'.format(extrm_var)
    elif sigma == -1:
        save_name = 'LabSea_JFM_climo_lower_{0}.mat'.format(extrm_var)
    met.save('/Volumes/Lovelace/PWP_data/LabSea_deep_conv_met/', save_name)


def LabSea_deep_conv_met(t_start, t_end, save_name=''):

    dat_root = '/Volumes/KatherineJ/ERA5_LabSea_dat/DeepConvZone_ts/{0}_1hr/'

    # 1: mu+std , 0: mu , -1 mu-std
    sigma = 0.0

    met = Meteorology(t_start, t_end, 1.0)

    #  get a list of yyyy_mm months to read & concatenate data
    months = []
    d_time = t_start
    while d_time < t_end:
        months.append(d_time.strftime('%Y_%m'))
        print(months[-1])
        d_time += relativedelta(months=1)

    variables = {
        'lw': 'msnlwrf',
        'sw': 'msnswrf',
        'qlat': 'mslhf',
        'qsens': 'msshf',
        'tx': 'ewss',
        'ty': 'nsss',
        'u': 'u10',
        'v': 'v10',
        'p': 'mtpr'
    }


    for i in variables.keys():
        var = variables[i]
        print(i, var)

        general_fname = 'era5_' + var +'_{0}.mat'
        all_dat = concat_data(dat_root.format(var), general_fname, months)
        dat = np.squeeze(all_dat[var])

        if var == 'mtpr':
            # print('skipping precip')
            # dat = dat*0
            dat = dat * 86400. # convert to mm/day

        if var == 'ewss' or var == 'nsss':
            # surface stress has units N m**-2 s
            dat = dat / 3600.  # convert to N/m^2


        # replace name string with actual time series
        variables[i] = dat


    met.set_wind_speed(variables['u'], variables['v'])
    met.set_wind_stress(variables['tx'], variables['ty'])
    met.set_heat_flux(variables['qlat'], variables['qsens'])
    met.set_radiation(variables['lw'], variables['sw'])
    met.set_precipitation(variables['p'])

    if save_name != '':
        met.save('/Volumes/Lovelace/PWP_data/LabSea_deep_conv_met/', save_name)
    return met
def add_const_tracer(met):
    # add const passive tracer
    N = len(met.time)

    mm_air = 0.029  # kg/mol
    rho_air = 1.2  # kg/m^3

    # add const oxygen
    mm_o2 = 0.032  # kg/mol
    ppm_o2 = 209460.  # ppm
    conc_o2 = rho_air / mm_air * ppm_o2 / 1.0e6 * mm_o2
    met.o2 = np.ones(N) * conc_o2

    #  add const CO2
    mm_co2 = 0.044  # kg/mol
    ppm_co2 = 400  # ppm
    conc_co2 = rho_air / mm_air * ppm_co2 / 1.0e6 * mm_co2
    met.co2 = np.ones(N) * conc_co2

    # print('o2:', conc_o2)
    # print('co2:', conc_co2)

    return met

def _compute_climo(var, y_start, y_end, month):
    '''
    helper function for climatological_met2()
    computes mean ts for given variable for years specified
    and one given month
    :param var: variable name (e.g. 'mslp')
    :param y_start: start year for climo (int)
    :param y_end: end year for climo (int)
    :param month: month number (int)
    :return: 1D array containing that month's climatology between specified years
    '''
    root = '/Volumes/KatherineJ/ERA5_LabSea_dat/DeepConvZone_ts/{0}_1hr/'.format(var)

    rgx = '(\d{4})_(\d{2})'
    all_ts = []
    n_min = np.inf
    for i in os.listdir(root):
        m = re.search(rgx, i)
        if not m:
            continue
        yr = int(m.group(1))
        mth = int(m.group(2))
        if mth == month and yr >= y_start and yr <= y_end:
            dat = sio.loadmat(root + i, squeeze_me=True)
            all_ts.append(dat[var])
            N = len(all_ts[-1])
            # if month in [10, 12, 1, 3] and N != 744:
            #     print(i)
            # elif month == 11 and N != 720:
            #     print(i)
            # elif month == 2 and N not  in [696, 672]:
            #     print(i)
            # print(N)
            if N < n_min:
                n_min = N

    all_ts = [i[:n_min] for i in all_ts]
    all_ts = np.array(all_ts)
    return np.mean(all_ts, axis=0)


def climatological_met2():
    '''
    Computes the net climatology from specified start and end
    :return: saves a .mat file compatible with use in PWP
    '''

    save_path = '/Volumes/Lovelace/PWP_data/meteorologies/'
    savename = 'LabSea_Oct-Mar_1950-2020_climo.mat'
    data_path = '/Volumes/KatherineJ/ERA5_LabSea_dat/DeepConvZone_ts/'

    start_year = 1951
    end_year = 2020
    start_month = 10
    end_month = 3

    '''
    initialise met object - chose 2019 because it's not a fucking leap year
    year doesn't matter since it's a climatological mean
    '''
    t_start = dt.datetime(2018, start_month, 1, 0, 0, 0)
    # t_end = dt.datetime(2019, end_month, 31, 23, 59, 59)
    t_end = dt.datetime(2019, end_month + 1, 1)
    met = Meteorology(t_start, t_end, 1.0)

    # keys are params of Met object, values are corresponding ERA5 data names
    met_vars = {'lw':'msnlwrf',
                'sw':'msnswrf',
                'qlat':'mslhf',
                'qsens':'msshf',
                'precip':'mtpr',
                'u':'u10',
                'v':'v10',
                'tx':'ewss',
                'ty':'nsss'}

    climo = {'lw':[],
                'sw':[],
                'qlat':[],
                'qsens':[],
                'precip':[],
                'u':[],
                'v':[],
                'tx':[],
                'ty':[]}

    for k in met_vars.keys():
        var = met_vars[k]
        print(var)
        now = t_start
        while now < t_end:
            print(now)
            ts = _compute_climo(var, start_year, end_year, now.month)
            climo[k].append(ts)
            now += relativedelta(months=1)

    for k in climo.keys():
        a = np.hstack(climo[k])
        climo[k] = a

    met.set_precipitation(climo['precip'] / 1000.)
    met.set_wind_speed(climo['u'], climo['v'])
    met.set_wind_stress(climo['tx'] / 3600., climo['ty'] / 3600.)
    met.set_radiation(climo['lw'], climo['sw'])
    met.set_heat_flux(climo['qlat'], climo['qsens'])
    met.save(save_path, savename)

    print('hello world')



if __name__ == '__main__':

    # make_test_met()
    # ERA5_met()
    # climatological_met(-1.0, 'precip')
    climatological_met2()

    # for yr in range(1979, 1980):
    #     # yr = 2015
    #     print(yr)
    #     start_date = dt.datetime(yr -1, 10, 1)
    #     end_date = dt.datetime(yr, 4, 1)
    #     met = LabSea_deep_conv_met(start_date, end_date)
    #     # met.plot_heat_winds()
    #
    #     met = add_const_tracer(met)
    #
    #
    #     save_name = 'LabSea_{0}_{1}.mat'.format(start_date.strftime('%Y-%m'), end_date.strftime('%Y-%m'))
    #     met.save('/Volumes/Lovelace/PWP_data/LabSea_deep_conv_met/LabSea_FW_const_tracers/', save_name)





