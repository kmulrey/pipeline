import numpy as np
import datetime
from scipy.interpolate import interp1d
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import cmath

coefficients_lba=([ 0.01489468, -0.00129305,  0.00089477, -0.00020722, -0.00046507],
                                        [ 0.01347391, -0.00088765,  0.00059822,  0.00011678, -0.00039787])



def calibrate(timestamp,data,cleaned_power,median_power):  # data array (dipoles,trace)

    frequencies = np.fft.rfftfreq(len(data[0]), d=5e-9)
    nantennas=len(data)
    #prepare galactic calibration  ADC -> Volts
    Calibration_curve = np.zeros(101)
    Calibration_frequencies=np.arange(101)*1e6
    Calibration_curve[29:82] = np.array([0, 1.37321451961e-05,1.39846332239e-05, 1.48748993821e-05, 1.54402170354e-05,1.60684568225e-05, 1.66241942741e-05, 1.67039066047e-05, 1.74480931848e-05,1.80525736486e-05,1.87066855054e-05, 1.88519099831e-05,1.99625051386e-05, 2.01878566584e-05,2.11573680797e-05,2.15829455528e-05,2.20133824866e-05,2.23736319125e-05, 2.24484419697e-05, 2.37802483891e-05, 2.40581543111e-05,2.42020383477e-05,2.45305869187e-05, 2.49399905965e-05,2.63774023804e-05,2.70334253414e-05, 2.78034857678e-05, 3.07147991391e-05, 3.40755705892e-05, 3.67311849851e-05,3.89987440028e-05,3.72257913465e-05, 3.54293510934e-05,3.35552370942e-05,2.96529815929e-05, 2.79271252352e-05, 2.8818544973e-05, 2.92478843809e-05,2.98454768706e-05, 3.07045462103e-05,3.07210553534e-05,3.16442871206e-05,3.2304638838e-05,3.33203882046e-05,3.46651060935e-05,3.55193137077e-05,3.73919275937e-05,3.97397037914e-05,4.30625048727e-05,4.74612081994e-05,5.02345866124e-05,5.53621848304e-05,0])
    f = interp1d(Calibration_frequencies, Calibration_curve)
    Cal_apply = f(frequencies)

    FFT_data=np.fft.rfft(data)
    
    #this is the normalization thing I don't understand
    
    dt_object = datetime.datetime.utcfromtimestamp(timestamp)
    observing_location = EarthLocation(lat=52.914921*u.deg, lon=6.869837540*u.deg)
    observing_time = Time(dt_object, scale='utc', location=observing_location)
    LST = observing_time.sidereal_time('mean').hour

    galactic_noise_power = (fourier_series(LST/(np.pi), coefficients_lba[0]), fourier_series(LST/(np.pi), coefficients_lba[1]))

    channel_width=(frequencies[1]-frequencies[0])
    scale=np.zeros([nantennas])
    scale[::2]=cleaned_power[::2]/median_power*galactic_noise_power[0] *channel_width/24.0
    scale[1::2]=cleaned_power[1::2]/median_power*galactic_noise_power[0] *channel_width/24.0
    FFT_data_cal=np.zeros_like(FFT_data)
    for i in np.arange(nantennas):
        FFT_data_cal[i]=FFT_data[i]*scale[i]
        FFT_data_cal[i]=FFT_data_cal[i]*Cal_apply
    
    data_calibrated=np.fft.irfft(FFT_data_cal)
    
    return data_calibrated





def fourier_series(x,p):
    """Evaluates a partial Fourier series.. math::
    F(x) \\approx \\frac{a_{0}}{2} + \\sum_{n=1}^{\\mathrm{order}} a_{n} \\sin(nx) + b_{n} \\cos(nx)
    """
    r = p[0] / 2
    order = int((len(p) - 1) / 2)
    for i in range(order):
        n = i + 1
        r += p[2*i + 1] * np.sin(n * x) + p[2*i + 2] * np.cos(n * x)
    return r


def DelayToPhase(frequencies,delays):
    # this is new--- needs to be checked
    T=1/frequencies
    cycle_fraction=np.zeros([len(delays),len(frequencies)])
    phase_shift=np.zeros([len(delays),len(frequencies)])
    weight=np.zeros([len(delays),len(frequencies)],dtype=complex)

    for i in np.arange(len(delays)):
        cycle_fraction[i]=delays[i]/T
        phase_shift[i]=cycle_fraction[i]*2*np.pi
        for k in np.arange(len(frequencies)):
            a=1*np.cos(phase_shift[i][k])
            b=1*np.sin(phase_shift[i][k])
            weight[i][k]=complex(a,b)
    return weight


def do_phase_correction(data,frequencies,cabledelays):
    #based on what is done in cr_physics.py

    data_fft=np.fft.rfft(data)
    weights=np.zeros([len(data_fft),len(data_fft[0])],dtype=complex)
    phases=np.zeros([len(data_fft),len(data_fft[0])],dtype=float)

    weights=DelayToPhase(frequencies,cabledelays)

    fft_data_new=data_fft*weights
    data_new=np.fft.irfft(fft_data_new)
    
    return data_new
