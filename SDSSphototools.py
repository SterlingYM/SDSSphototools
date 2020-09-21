import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import simple_norm
import astropy.units as u
from photutils import SkyCircularAperture, aperture_photometry


# from tqdm.notebook import tqdm
# tqdm.pandas()
import os
import glob
import bz2
from bs4 import BeautifulSoup
import urllib.request


# SDSS-unique parameters
f0 = 10**(22.5/2.5) # such that pogson_mag(f0)=0
asinh_mag_params = { 
    # from http://www.sdss3.org/dr8/algorithms/magnitudes.php#asinh_table
    'u': 1.4e-10,
    'g': 0.9e-10,
    'r': 1.2e-10,
    'i': 1.8e-10,
    'z': 7.4e-10
}

def pogson_mag(flux):
    '''
    returns magnitude from SDSS flux with 'conventional' log scale
    m = 22.5 - 2.5log10(f)
    '''
    return 22.5 - 2.5 * np.log10(flux)

def asinh_mag(flux,filt,f0=10**(22.5/2.5)):
    '''
    returns magnitude in asinh scale calculted from SDSS flux
    TODO: check f0 value
    '''
    b = asinh_mag_params[filt]
    return -2.5/np.log(10) * (np.arcsinh((flux/f0)/(2*b))+np.log(b))

def do_photometry_radec(RA,DEC,z,r,name='unnamed',base_path='./.tmp/',show_plots=False):
    '''
    A tool to perform aperture photometry at given RA & Dec.
    The sum of flux in each pixel within the aperture is used to calulate the magnitude.
    
    Args:
        RA: Right Ascension in degrees (J2000)
        DEC: Declination of the object in degrees (J2000)
        z: Redshift of the object. Used to determine the projected angular size of aperture.
        r: Aperture object with Astropy units. (e.g. 2*u.kpc)
        name: name of the object.
        base_path: the directory in which all downloaded data are stored.
    
    Returns:
        files: a list of image files downloaded from SDSS.
        magdata: a pandas dataframe cotaining estimated magnitudes.
    '''
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    # retrieve page
    print('* querying data...')
    page = urllib.request.urlopen(f"https://dr12.sdss.org/fields/raDec?ra={RA}&dec={DEC}")
    soup = BeautifulSoup(page, 'html.parser')

    # iterate through page and download relevant files
    print('* downloading image files...')
    files_compressed = []
    for ss in soup.find_all('a', href=True):
        if 'bz2' in str(ss):
            filt = ss['href'].split('frame-')[1][0]
            outfile = '{}{}-{}.fits.bz2'.format(base_path, name, filt)
            urllib.request.urlretrieve("https://dr12.sdss.org{}".format(ss['href']), outfile)
            files_compressed.append(outfile)
            
    # decompress files
    print('* decompressing image files...')
    files = []
    for filepath in files_compressed:
        zipfile = bz2.BZ2File(filepath) # open the file
        newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
        data = zipfile.read() # get the decompressed data
        open(newfilepath, 'wb').write(data) # write a uncompressed file
        files.append(newfilepath)
            
    # determine aperture size 
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    Mpc_per_rad = cosmo.angular_diameter_distance(z)
    radius_arcsec = (r.to(u.Mpc) / Mpc_per_rad * u.rad).to(u.arcsec)
    pos = SkyCoord(RA * u.deg, DEC * u.deg)

    # do photometry
    print('* performing photometry...')
    filters = []
    mags_asnh = []
    mags_pogson = []
    for fl in files:
        with fits.open(fl, memmap = True) as img:
            img_data = img[0].data
            cs = WCS(header = img[0].header)
            filt = img[0].header['FILTER']

            # photometry
            aperture = SkyCircularAperture(pos, radius_arcsec).to_pixel(cs)
            local_flux = aperture.to_mask().multiply(img_data)
            mean_flux  = local_flux.flatten().sum()
            mag1 = asinh_mag(mean_flux,filt) # asinh
            mag2 = pogson_mag(mean_flux) # pogson
            mags_asnh.append(mag1)
            mags_pogson.append(mag2)
            filters.append(filt)

            # plot
            if show_plots:
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6.5))
                norm = simple_norm(img_data, 'sqrt', percent=99)
                im = ax1.imshow(img_data,norm=norm)
                aperture.plot(color='red', lw=2,axes=ax1)
                ax2.imshow(local_flux)
                plt.colorbar(im)
            
    # data
    magdata = pd.DataFrame(columns=['u','g','r','i','z'])
    for filt,mag1,mag2 in zip(filters,mags_asnh,mags_pogson):
        magdata.loc['asinh',filt] = mag1
        magdata.loc['pogson',filt] = mag2
    print(magdata)
    
    return files, magdata