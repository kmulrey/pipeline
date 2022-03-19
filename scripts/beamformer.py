import numpy as np
import cmath
from scipy.optimize import fmin_powell
import pipeline_helper as helper
lightspeed=299792458.0


def GeometricDelayFarField(position, direction, length):
    delay=(direction[0]*position[0] + direction[1]*position[1]+direction[2]*position[2])/length/lightspeed
    return delay


def minibeamformer(fft_data,frequencies,positions,direction):
    #adapted from hBeamformBlock
    nantennas=len(positions)
    nfreq=len(frequencies)
    output=np.zeros([len(frequencies)],dtype=complex)

    norm = np.sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2])
    
    for a in np.arange(nantennas):
        delay = GeometricDelayFarField(positions[a], direction, norm)
        #print(delay)
    
        for j in np.arange(nfreq):
            real = 1.0 * np.cos(2*np.pi*frequencies[j]*delay)
            imag = 1.0 * np.sin(2*np.pi*frequencies[j]*delay)
            de=complex(real,imag)
            output[j]=output[j]+fft_data[a][j]*de
              #*it_out += (*it_fft) * polar(1.0, (2*np.pi)*((*it_freq) * delay));
    
    return output


def geometric_delays(antpos,sky):
    distance=np.sqrt(sky[0]**2+sky[1]**2+sky[2]**2)
    delays=(np.sqrt((sky[0]-antpos[0])**2+(sky[1]-antpos[1])**2+(sky[2]-antpos[2])**2)-distance)/lightspeed
    return delays
    
    
def beamformer(fft_data,frequencies,delay):
    nantennas=len(delay)
    nfreq=len(frequencies)
    output=np.zeros([len(frequencies)],dtype=complex)

    for a in np.arange(nantennas):
        for j in np.arange(nfreq):
            real = 1.0 * np.cos(2*np.pi*frequencies[j]*delay[a])
            imag = 1.0 * np.sin(2*np.pi*frequencies[j]*delay[a])
            de=complex(real,imag)
            output[j]=output[j]+fft_data[a][j]*de
    return output

def directionFitBF(fft_data,frequencies,antpos,start_direction,maxiter):
    
    def negative_beamed_signal(direction):
        rho=1.0
        theta=(2*np.pi) - np.radians(direction[1])
        phi=(2*np.pi) - np.radians(direction[0])
        direction_cartesian=helper.spherical2cartesian(rho,theta, phi)
        delays=geometric_delays(antpos,direction_cartesian)
        out=beamformer(fft_data,frequencies,delays)
        timeseries=np.fft.irfft(out)
        return -100*np.max(timeseries**2)
    
    
    fit_direction = fmin_powell(negative_beamed_signal, np.asarray(start_direction), maxiter=maxiter, xtol=1.0)
    
    rho=1.0
    theta=(2*np.pi) - np.radians(fit_direction[1])
    phi=(2*np.pi) - np.radians(fit_direction[0])
    direction_cartesian=helper.spherical2cartesian(rho,theta, phi)
    delays=geometric_delays(antpos,direction_cartesian)
    out=beamformer(fft_data,frequencies,delays)
    timeseries=np.fft.irfft(out)
    
    
    return fit_direction, timeseries



def return_minibeamformed_data(timeseries_data,positions,direction):


    fft_data_0=np.fft.rfft(timeseries_data)[::2]
    fft_data_1=np.fft.rfft(timeseries_data)[1::2]
    timeseries_0=timeseries_data[0::2]
    timeseries_1=timeseries_data[1::2]
    frequencies=np.fft.rfftfreq(len(timeseries_data[0]), d=5e-9)


    positions_0=positions[::2]
    positions_1=positions[1::2]
    x,y,z=helper.spherical2cartesian(1,(np.pi / 2) - np.radians(direction[1]),(np.pi / 2) - np.radians(direction[0]))
    direction_cartesian=np.array([x,y,z])
    
    beamed_fft_0=minibeamformer(fft_data_0,frequencies,positions_0,direction_cartesian)
    beamed_fft_1=minibeamformer(fft_data_1,frequencies,positions_1,direction_cartesian)
    
    beamformed_timeseries_0=np.fft.irfft(beamed_fft_0)
    beamformed_timeseries_1=np.fft.irfft(beamed_fft_1)
    
    return beamformed_timeseries_0,beamformed_timeseries_1
