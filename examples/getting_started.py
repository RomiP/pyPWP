# Import the PWP model and input file classes.
from pwp.pwp import PWP
from pwp.inputs.meteorology import Meteorology
from pwp.inputs.profile import Profile

# other imports
import datetime as dt
import numpy as np
import pandas as pd
import seawater as sw
import xarray as xr


if __name__ == '__main__':
    '''
    Before we  run the model, we need to create the meteorology and initial profile objects. 
    
    Here, we'll walk through how to create each object and set up the model. I've included sample data for the
    Labrador Sea (Oct 2008 - Mar 2009) stored as CSV files
    '''


    # ------------------------------------------------------
    # How to make a Meteorology file for PWP

    '''
    The PWP model calculates fluxes of heat, salinity, and momentum. We therefore need to 
    create an input file containing atmoshperic conditions to force the model
    '''

    met_data_cvs = './LabSea_sample_data/met_data.csv'
    met_in = pd.read_csv(met_data_cvs)

    # we can either read these from the file, or just set them manually
    start_date = dt.datetime(2008, 10 , 1)
    end_date = dt.datetime(2009, 4, 1)
    time_step = 6 # hours

    # let's read our data into a dict keyed by variable name recognised by Meteorology class
    time  = met_in['Time (days since 0000-01-01 00:00:00)'] * 24
    time = time - time[0]
    data_dict = {
        'time': time,
        'lw': met_in['Longwave (W/m^2)'],
        'sw': met_in['Shortwave (W/m^2)'],
        'qlat': met_in['Latent (W/m^2)'],
        'qsens': met_in['Sensible (W/m^2)'],
        'tx': met_in['Wind Stress u (N/m^2)'],
        'ty': met_in['Wind Stress v (N/m^2)'],
    }
    # Note: we don't need to specify everything. Missing fields are set to 0

    # create the meteorology object
    met = Meteorology(start_date, end_date, time_step, data_dict)

    # # You can choose to save this object to use later
    # met.save('path/to/destination'
    # # you can read in your saved object like this:
    # met = Meteorology(start_date, end_date, time_step)
    # met.read_from_file('path/to/saved/file')

    '''
    Now let's read in some data to use as an initial profile
    We'll create a Profile object for PWP using argo measurements
    '''

    argo_file = './LabSea_sample_data/2008_10_24|10_30_58|58.1N|-52.9E|D4901446_029.nc'
    argoxr = xr.open_dataset(argo_file)
    latitude = float(argoxr['LATITUDE'])

    # temp, sal, depth, lat,
    depth = sw.dpth(np.squeeze(argoxr['PRES_ADJUSTED']), latitude)
    prof = Profile(
        np.squeeze(argoxr['TEMP_ADJUSTED']),
        np.squeeze(argoxr['PSAL_ADJUSTED']),
        depth,
        latitude
    )

    # let's plot the density profile to assess stability
    prof.plot_profile('d', xlabel='Density (kg/m^3)')

    '''
    PWP mixes to relieve static instabilities, this means the profile will be stable 
    after each time step. Let's remove any static instabilities in this initial profile
    so the model doesn't immediately mix more of the water column than necessary
    '''

    prof.remove_si(50) # skip the mixed later (50 corresponds to index, not depth)
    prof.plot_profile('d', xlabel='Density (kg/m^3)', title='Removed Static Instabilities')

    # # likewise, we can save this profile for later use
    # prof.save('path/to/destination')
    # # then read it back in with the following
    # from pwp.inputs.profile import build_from_existing
    # prof = build_from_existing('path to saved file')

    '''
    Now that we have our meteorology and initial profile set up, we can initialise the model
    '''

    myPWP = PWP()
    myPWP.diffusion_ON = False # the diffusion parameterisation is a bit dodgy.
    myPWP.max_depth = 990 # do this before reading in data
    myPWP.read_in_init_data(prof, met)
    myPWP.run(save_name='./test_run.nc', write_info=True)




