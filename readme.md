# SDSSphototools

Yukei S. Murakami (sterling.astro@berkeley.edu)

A set of helper tools to perform aperture photometry on SDSS images.

----------------------
## Requirements
Dependencies and required python packages

* python3 (tested w/ 3.7)
* numpy
* matplotlib
* pandas
* astropy
* photutils
* os
* glob
* bz2
* bs4
* urllib

TODO: add ```requirements.txt```

## usage
~~~~.python
import SDSSphototools as spt
import astropy.units as u

RA = 359.738
Dec = -2.248
z = 0.0254
r = 1.5 * u.kpc
name = 'SN2003he_host'

files,magdata = spt.do_photometry_radec(RA,Dec,z,r,name=name,show_plots=True)
~~~~
