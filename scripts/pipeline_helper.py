import sys
import numpy as np
from os import mkdir
from os.path import isdir
from pickle import dump
import imp
import os
from datetime import datetime
import glob, os
import sys


event_id_offset=1262304000
radio_directory='/vol/astro3/lofar/vhecr/lora_triggered/data/'
particle_directory='/vol/astro3/lofar/vhecr/lora_triggered/LORA/'



# return filename by UTC
def return_file_by_UTC(timestamp):

    dt_object = datetime.utcfromtimestamp(timestamp)
    year=dt_object.year
    month=dt_object.month
    day=dt_object.day
    hour=dt_object.hour
    minute=dt_object.minute
    sec=dt_object.second
    

    radio_file_tag='D'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'T'+str(hour).zfill(2)+str(minute).zfill(2)+str(sec).zfill(2)
    particle_file_tag='LORAdata-'+str(dt_object.year)+str(dt_object.month).zfill(2)+str(dt_object.day).zfill(2)+'T'+str(dt_object.hour).zfill(2)+str(dt_object.minute).zfill(2)+str(dt_object.second).zfill(2)+'.dat'

    return radio_file_tag,particle_file_tag

# return event ID from UTC
def ID_from_UTC(timestamp):
    return timestamp-event_id_offset
    
# return event ID from YMDHMS, aka LOFARFILENAME
#note- this has some issue, I think with an hour offset
def ID_from_YMDHMS(YMDHMS):
    tmp=list(YMDHMS)
    year=int(tmp[1]+tmp[2]+tmp[3]+tmp[4])
    month=int(tmp[5]+tmp[6])
    day=int(tmp[7]+tmp[8])
    hour=int(tmp[10]+tmp[11])
    minute=int(tmp[12]+tmp[13])
    second=int(tmp[14]+tmp[15])
    return int(datetime(year,month,day,hour,minute,second).timestamp()-1262304000)


# return filename by LOFAR_ID
def return_file_by_LOFAR_ID(id):
    timestamp=id+event_id_offset
    dt_object = datetime.utcfromtimestamp(timestamp)
    year=dt_object.year
    month=dt_object.month
    day=dt_object.day
    hour=dt_object.hour
    minute=dt_object.minute
    sec=dt_object.second

    radio_file_tag='D'+str(year)+str(month).zfill(2)+str(day).zfill(2)+'T'+str(hour).zfill(2)+str(minute).zfill(2)+str(sec).zfill(2)
    particle_file_tag='LORAdata-'+str(dt_object.year)+str(dt_object.month).zfill(2)+str(dt_object.day).zfill(2)+'T'+str(dt_object.hour).zfill(2)+str(dt_object.minute).zfill(2)+str(dt_object.second).zfill(2)+'.dat'

    return radio_file_tag,particle_file_tag


def return_station_files(file_tag):
    radio_file=[]
    os.chdir(radio_directory)
    for file in glob.glob('*'+file_tag+'*'):
        radio_file.append(radio_directory+file)
    return radio_file

def return_stations(file_list):
    stations=[]
    for i in np.arange(len(file_list)):
        stations.append((file_list[i].split('_')[3]))
    
    return stations

def return_lora_file(file_tag):
    os.chdir(particle_directory)
    for file in glob.glob(file_tag):
        lora_file=particle_directory+file

    return lora_file


def return_lora_direction(file):
    data=np.genfromtxt(file,comments='//',skip_footer=20)
    elevation=float(data[4])
    azimuth=float(data[5])
    return (azimuth,elevation)



def return_lora_time(file):
    #cr_physics version chechs +/- time
    data=np.genfromtxt(file,comments='//',skip_footer=20)
    utc=float(data[0])
    nsec=float(data[1])
    return utc,nsec

def loraTimestampToBlocknumber(lora_seconds, lora_nanoseconds, starttime, samplenumber, clockoffset=1e4, blocksize=2 ** 16,samplingfrequency=200):
    #katie- what is clockoffset??
    """Calculates block number corresponding to LORA timestamp and the
    sample number within that block (i.e. returns a tuple
    (``blocknumber``,``samplenumber``)).

    Input parameters:

    =================== ==============================
    *lora_seconds*      LORA timestamp in seconds (UTC timestamp, second after 1st January 1970).
    *lora_nanoseconds*  LORA timestamp in nanoseconds.
    *starttime*         LOFAR_TBB timestamp.
    *samplenumber*      Sample number.
    *clockoffset*       Clock offset between LORA and LOFAR.
    *blocksize*         Blocksize of the LOFAR data.
    =================== ==============================

    """

    lora_samplenumber = (lora_nanoseconds - clockoffset) * samplingfrequency*1e-3 #MHz to nanoseconds

    value = (lora_samplenumber - samplenumber) + (lora_seconds - starttime) * samplingfrequency*1e6

    if value < 0:
        raise ValueError("Event not in file.")

    return (int(value / blocksize), int(value % blocksize))



def spherical2cartesian(rho,theta, phi):
    #adapted from tmf_spherical2cartesian in cartesian_spherical.c
    st = np.sin(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    ct = np.cos(theta)
    x = rho * st * cp
    y = rho * st * sp
    z = rho * ct
    return x,y,z
