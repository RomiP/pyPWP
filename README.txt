# PWP_python

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

******************************************************************************

The required input files should be structured as follows
  -> The positive direction is taken to be from atmosphere to ocean

PROFILE:
  fields (* mandatory, - optional)
    * 'z' depth vector [m]
    * 's' salinity [psu]
    * 't' temperature [deg C]
    * 'lat' latitude [deg E]
    - 'lon' longitude [deg N]
    - tracers [kg/m^3]

METEOROLOGY:
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
      - tracers (atmospheric concentration) [kg/m^2]
      - 't_ref' reference time & date (string)
      

******************************************************************************

The model is initialized with the following default settings. All of these can
be modified prior to running the model.

  True/False flag to turn ON/OFF forcing.
    winds_ON = True # winds
    emp_ON = True # freshwater
    heat_ON = True # heat flux
    drag_ON = True # current drag due to internal-inertial wave breaking
    diffusion_ON = True # background molecular diffusion

  model settings
    dt = 3600. # time step [seconds]
    dz = 1.0 # depth increment [meters]
    max_depth = 1000. # Max depth of vertical coordinate [meters]
    mld_thresh = 0.2 # temperature change criterion for MLD [deg C]

  physical constants
    rb = 0.65 # critical bulk richardson number [dimensionless]
    rg = 0.25  # critical gradient richardson number [dimensionless]
    rkz = 1.0e-5 # background vertical diffusion [m**2/s]
    lambda_red =  0.6 # longwave extinction coefficient [meters]
    lambda_blue = 20.  # shortwave extinction coefficient [meters]
