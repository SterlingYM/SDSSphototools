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
    returns magnitude from SDSS flux with 'conventional' log scale. flux is in nanomaggy.
    m = 22.5 - 2.5log10(f)
    '''
    return 22.5 - 2.5 * np.log10(flux)

def asinh_mag(flux_nano,filt):
    '''
    returns magnitude in asinh scale calculted from SDSS flux. Note that 
    Note that convension of 'flux' is different from 'f' in pogson_mag (SDSS website is very confusing indeed).
    '''
    b = asinh_mag_params[filt]
    flux = flux_nano * 1e-9
    return -2.5/np.log(10) * (np.arcsinh(flux/(2*b))+np.log(b))

def sdss_download_fits(RA,DEC,base_path='./.tmp',verbal=False,name='unnamed'):
    '''
    Download image files from SDSS. 
	 Target image files contain RA and DEC given and are returned as fits files.
	 Fits files contain flux info, in units of nanomaggies.
	 
	 Args:
		  RA: Right Ascension of the object in degrees (J2000)
		  Dec: Declination of the oeject in degrees (J2000)
		  base_path: the directory in which all downloaded data are stored
		  verbal: switch to turn on/off progress report
	 Returns:
		  files_compressed: a list of image files downloaded from SDSS. Each file is compressed by bz2.
    '''
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    # retrieve page
    if verbal:
        print('* querying data...',end='')
    page = urllib.request.urlopen(f"https://dr12.sdss.org/fields/raDec?ra={RA}&dec={DEC}")
    soup = BeautifulSoup(page, 'html.parser')
    if verbal:
	    print('Done')
		
    # iterate through page and download relevant files
    if verbal:
        print('* downloading image files...',end='')
    files_compressed = []
    for ss in soup.find_all('a', href=True):
        if 'bz2' in str(ss):
            filt = ss['href'].split('frame-')[1][0]
            outfile = '{}{}-{}.fits.bz2'.format(base_path, name, filt)
            urllib.request.urlretrieve("https://dr12.sdss.org{}".format(ss['href']), outfile)
            files_compressed.append(outfile)
    if verbal:
        print('Done')
    return files_compressed
	
def decompress_bz2(files_compressed,verbal=False):
    '''
    Decompresses bz2 compressed files.
    '''
    if verbal:
        print('* decompressing image files...',end='')
    files = []
    for filepath in files_compressed:
        zipfile = bz2.BZ2File(filepath) # open the file
        newfilepath = filepath[:-4] # assuming the filepath ends with .bz2
        data = zipfile.read() # get the decompressed data
        open(newfilepath, 'wb').write(data) # write a uncompressed file
        files.append(newfilepath)
    if verbal:
        print('Done')
    return files

def angular_dist(r,z,d=None,H0=70,Om0=0.3):
    '''
    Calculates projected angular distance of given local size r at redshift z.
    TODO: add option to calculate from distance rather than z
    '''
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    Mpc_per_rad = cosmo.angular_diameter_distance(z)
    theta = (r.to(u.Mpc) / Mpc_per_rad * u.rad)
    return theta

def do_photometry_radec(RA,DEC,r=None,z=None,theta=None,files=None,name='unnamed',base_path='./.tmp/',show_plots=False,verbal=False):
    '''
    A tool to perform aperture photometry at given RA & Dec.
    The sum of flux in each pixel within the aperture is used to calulate the magnitude.

    Args:
        RA: Right Ascension in degrees (J2000)
        DEC: Declination of the object in degrees (J2000)
        r: Aperture object with Astropy units. (e.g. 2*u.kpc)
        z: Redshift of the object. Used to determine the projected angular size of aperture.
    theta: Aperture angular radius with Astropy units (e.g. 2*u.arcsec). r and z are ignored if this is given.
    files: A list of paths (filenames) to sdss fits files. Files won't be newly downloaded if this is given. RA and DEC are still required for photometry.
        name: name of the object.
        base_path: the directory in which all downloaded data are stored.
    Returns:
        files: a list of image files downloaded from SDSS.
        magdata: a pandas dataframe cotaining estimated magnitudes.
    '''
	# file download
    if files == None:
        files_compressed = sdss_download_fits(RA,DEC,base_path,verbal=verbal,name=name)
        files = decompress_bz2(files_compressed,verbal=verbal)
            
    # determine aperture size from r and z
    if theta == None:
        if (r==None) or (z==None):
            print('local radius r and redshift z are required.')
        theta = angular_dist(r,z)
    pos = SkyCoord(RA * u.deg, DEC * u.deg)
    aperture_obj = SkyCircularAperture(pos, theta.to(u.arcsec))
	
    # do photometry
    if verbal:
        print('* performing photometry...',end='')
    filters = []
    mags_asnh = []
    mags_pogson = []
    for fl in files:
        with fits.open(fl, memmap = True) as img:
            img_data = img[0].data
            cs = WCS(header = img[0].header)
            filt = img[0].header['FILTER']

            # photometry
            aperture = aperture_obj.to_pixel(cs)
            local_flux = aperture.to_mask().multiply(img_data)
            total_flux  = local_flux.flatten().sum()
            mag1 = asinh_mag(total_flux,filt) # asinh
            mag2 = pogson_mag(total_flux) # pogson
            mags_asnh.append(mag1)
            mags_pogson.append(mag2)
            filters.append(filt)

            # plot
            if show_plots:
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6.5))
                norm = simple_norm(img_data, 'sqrt', percent=99)
                im1 = ax1.imshow(img_data,norm=norm)
                plt.colorbar(im1,ax=ax1)
                aperture.plot(color='red', lw=2,axes=ax1)
                im2 = ax2.imshow(local_flux)
                plt.colorbar(im2,ax=ax2)
    if verbal:
        print('Done')
		
    # data
    magdata = pd.DataFrame(columns=['u','g','r','i','z'])
    for filt,mag1,mag2 in zip(filters,mags_asnh,mags_pogson):
        magdata.loc['asinh',filt] = mag1
        magdata.loc['pogson',filt] = mag2
    if verbal:
        print(magdata)
    
    return files, magdata