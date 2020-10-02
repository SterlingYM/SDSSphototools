import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import simple_norm
import astropy.units as u
from photutils import SkyCircularAperture, aperture_photometry

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
gain_old = pd.read_csv('gain_old.csv')
gain_new = pd.read_csv('gain_new.csv')
dvar_old = pd.read_csv('dvar_old.csv')
dvar_new = pd.read_csv('dvar_new.csv')

##### Fits file preparation tools #####
def get_filter(hdu):
    filt = hdu[0].header['FILTER']
    return filt

def get_run(hdu):
    run = hdu[0].header['RUN']
    return run

def get_camcol(hdu):
    camcol = hdu[0].header['CAMCOL']
    return camcol

def get_gain(hdu):
    camcol = get_camcol(hdu)
    filt = get_filter(hdu)
    run = get_run(hdu)
    if run<1100:
        return gain_old.loc[camcol,filt]
    else:
        return gain_new.loc[camcol,filt]

def get_dvar(hdu):
    '''Returns dark variance for given image'''
    camcol = get_camcol(hdu)
    filt = get_filter(hdu)
    run = get_run(hdu)
    if run<1500:
        return dvar_old.loc[camcol,filt]
    else:
        return dvar_new.loc[camcol,filt]

def get_img(hdu):
    img  = hdu[0].data
    return img

def get_cimg(hdu):
    ncol = hdu[0].data.shape[0]
    calib = hdu[1].data
    cimg = np.tile(calib,(ncol,1))
    return cimg

def get_simg(hdu,method='fast'):
    sky   = hdu[2].data
    allsky  = sky['allsky'][0]
    xinterp = sky['xinterp'][0]
    yinterp = sky['yinterp'][0]
    y,x = np.indices(allsky.shape)

    ## remap to original coordinate
    if method == 'accurate':
        # edge extrapolation is performed but is very slow
        f = interpolate.interp2d(x,y,allsky,kind='linear')
        simg = f(xinterp,yinterp)
    if method == 'fast':
        # edge extrapolation is not performed and nearest edge value is returned
        points = np.array([x.flatten(),y.flatten()]).T
        grid_x,grid_y = np.meshgrid(xinterp,yinterp)
        simg = interpolate.griddata(points, allsky.flatten(), (grid_x, grid_y), method='linear')#, fill_value=0)
        simg_edge = interpolate.griddata(points, allsky.flatten(), (grid_x, grid_y), method='nearest')
        simg[np.isnan(simg)]=simg_edge[np.isnan(simg)]
    return simg

def get_dn(hdu,method='fast'):
    img = get_img(hdu)
    cimg = get_cimg(hdu)
    simg = get_simg(hdu,method=method)
    dn= img/cimg+simg
    return dn

def get_err(hdu,method='fast'):
    '''Returns an array of flux errors in nanomaggies for given image'''
    dn = get_dn(hdu,method) # dn
    gain = get_gain(hdu) # gain
    dvar = get_dvar(hdu) # dark variance
    cimg = get_cimg(hdu)
    
    dn_err = np.sqrt(dn/gain + dvar)
    img_err = dn_err * cimg
    return img_err


##### Photometry Tools #####
def pogson_mag(flux,flux_err=None):
    '''
    returns magnitude from SDSS flux with 'conventional' log scale. flux is in nanomaggy.
    m = 22.5 - 2.5log10(f)
    sigma_m = -2.5 * f_err / (ln(10) * f)
    '''
    mag = 22.5 - 2.5 * np.log10(flux)
    if flux_err != None:
        err = -2.5 * flux_err / (np.log(10) * flux)
        return mag, err
    else:
        return mag

def asinh_mag(filt,flux_nano,flux_nano_err=None):
    '''
    Returns magnitude in asinh scale calculted from SDSS flux.
    Note that convension of 'flux' is different from 'f' in pogson_mag (SDSS website is very confusing indeed).
    input flux is in nanomaggies, flux_nano == F/1e-9.
    F = (f/f0) # f0 = zero point mag
    A = -2.5/ln(10)
    m = A * asinh(F/(2b)) + A * ln(b)
    sigma_m = (A sigma_F) / sqrt((F/(2b))^2 + 1)
    '''
    b = asinh_mag_params[filt]
    flux = flux_nano * 1e-9
    mag = -2.5/np.log(10) * (np.arcsinh(flux/(2*b))+np.log(b))
    if flux_nano_err != None:
        ferr = flux_nano_err * 1e-9
        err = -2.5/np.log(10) * ferr / np.sqrt( (flux/(2*b))**2 + 1)
        return mag, err
    else:
        return mag

def asinh_mag_inverse(filt,mag,mag_err):
    '''
    Returns flux in maggies (= nanomaggies * 1e-9) calculated from magnitudes.
    '''
    b = asinh_mag_params[filt]
    flux = np.sinh(mag/(2.5/np.log(10))-np.log(b)) * (2*b)
    flux_err = mag_err / -2.5/np.log(10) * np.sqrt( (flux/(2*b))**2 + 1)
    return flux,flux_err
    
##### SDSS Query Tools #####
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
    err_asinh = []
    err_pogson = []
    for fl in files:
        hdu = fits.open(fl, memmap = True)
        cs = WCS(header = hdu[0].header)
        img = get_img(hdu)
        err = get_err(hdu)
        filt = get_filter(hdu)

        # photometry
        aperture = aperture_obj.to_pixel(cs)
        local_flux = aperture.to_mask().multiply(img)
        local_err  = aperture.to_mask().multiply(err)
        total_flux = local_flux.flatten().sum()
        total_err  = np.sqrt((local_err**2).sum()) # quadrature
        mag1,err1 = asinh_mag(filt,total_flux,total_err) # asinh
        mag2,err2 = pogson_mag(total_flux,total_err) # pogson
        mags_asnh.append(mag1)
        mags_pogson.append(mag2)
        err_asinh.append(err1)
        err_pogson.append(err2)
        filters.append(filt)

        # plot
        if show_plots:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6.5))
            norm = simple_norm(img, 'sqrt', percent=99)
            im1 = ax1.imshow(img,norm=norm)
            plt.colorbar(im1,ax=ax1)
            aperture.plot(color='red', lw=2,axes=ax1)
            im2 = ax2.imshow(local_flux)
            plt.colorbar(im2,ax=ax2)
    if verbal:
        print('Done')
		
    # data
    magdata = pd.DataFrame(columns=['u','g','r','i','z','u_err','g_err','r_err','i_err','z_err'])
    for filt,mag1,mag2,err1,err2 in zip(filters,mags_asnh,mags_pogson,err_asinh,err_pogson):
        magdata.loc['asinh',filt] = mag1
        magdata.loc['pogson',filt] = mag2
        magdata.loc['asinh',filt+'_err'] = err1
        magdata.loc['pogson',filt+'_err'] = err2
    if verbal:
        print(magdata)
    
    return files, magdata