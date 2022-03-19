from pickle import load
import numpy as np
from scipy.signal import gaussian


class Event:
    def __init__(self, name):
        self.name = name  # LOFAR event ID
        #self.type = type  # V1 or V2
        self.lora_direction=(0,0)   # azimuth, elevation
    
    '''
    theta=0
    elevation=0
    phi=0
    fit_theta=0
    fit_elevation=0
    fit_phi=0
    fit_theta_err=0
    fit_phi_err=0
    x_core=0
    y_core=0
    x_core_err=0
    y_core_err=0
    z_core=0
    UTC_min=0
    nsec_min=0
    energy=0
    energy_err=0
    Rm=0
    fit_elevation_err=0
    fit_phi_err=0
    Ne=0
    Ne_err=0
    CorCoef_xy=0
    Ne_RefA=0
    NeErr_RefA=0
    Energy_RefA=0
    EnergyErr_RefA=0
    direction_flag=0
    event_flag=0
    '''



class Station:
    def __init__(self, name):
        self.name = name  # LOFAR event ID
        #self.type = type  # V1 or V2
        #self.lora_direction=(0,0)   # azimuth, elevation
        self.status='BAD'
        self.positions = [] 

    #outfile = "calibrated_pulse_block-{0}-{1}.npy".format(options.id, station.stationname)

    '''
    theta=0
    elevation=0
    phi=0
    fit_theta=0
    fit_elevation=0
    fit_phi=0
    fit_theta_err=0
    fit_phi_err=0
    x_core=0
    y_core=0
    x_core_err=0
    y_core_err=0
    z_core=0
    UTC_min=0
    nsec_min=0
    energy=0
    energy_err=0
    Rm=0
    fit_elevation_err=0
    fit_phi_err=0
    Ne=0
    Ne_err=0
    CorCoef_xy=0
    Ne_RefA=0
    NeErr_RefA=0
    Energy_RefA=0
    EnergyErr_RefA=0
    direction_flag=0
    event_flag=0
    '''
