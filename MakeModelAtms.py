# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 23:46:26 2022

@author: alice
"""

# This is for making model atmosphere files for stars. It's still a
# bit messy because it's just lines of code doing things, nothing's
# contained nicely in a function, but it really is just running one
# cell after another. You get coordinates, you match them with the 
# gaia catalog, restrict the matched coordinates by magnitude since
# we know what magnitudes to expect, do a bunch of calculations,
# and write all this data into the file 'gaia_m15_final.fits'

# Main issue is that several of the stars that got matched from gaia
# don't seem to be real members of M15. We're not sure why.

# In[Initialization]:

import StarSpec as ss
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.io import fits
from specutils.manipulation import SplineInterpolatedResampler, LinearInterpolatedResampler
splr = SplineInterpolatedResampler()
linr = LinearInterpolatedResampler()
import astropy.units as u
import matplotlib.pyplot as plt

# Global plot formatting
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

delims = r"[._]"

mpl.rcParams['agg.path.chunksize'] = 1000000

starnames = ['M15C29294_1631', 'HD204712', 'M15/K969', 'H204712', 
            'K386', 'K462', 'M15_K490', 'M15-C29523_1114', 
            'M15C31264_1605', 'k969 m15', 'm15c29413x1023', 
            'M15_I-63', 'm15bcf315', 
            'M15_III-59', 'C30131_0829', 'K731,I-63', 
            'K934,I-62', 'K479', 'K341', 'm15bcf203', 'K583', 
            'II-64 (Flux)', 'M15_K969', 'C30247_0911', 
            'M15C30599_2226', 'M15-G30482_1054', 'M15-pr109', 
            'M15C30027_0019', 'M56 I 22 (Flux)', 'M15/K431', 
            'M15C29589_1223', 'K582,IV-62', 'M15-K853', 'M15_I-62', 
            '1-38', 's3 (Flux)', 'M15_K431', 'M15_IV-62', 'K1084']

# These are ordered the same way as in starnames
exps = [['51990'], ['54353'], ['50889', '51435', '51981', '51347'], ['39707', '41219', '47488', '39833'], ['36335', '38324', '40315', '42303', '44614', '46610', '48599'], ['32465', '36476', '38471', '40465', '43119', '45186', '47180'], ['40897', '42463', '44030', '45596'], ['34706'], ['49605', '50700'], ['48271', '50181', '52090'], ['29313', '31183', '33277', '35138', '36994'], ['51794', '51664'], ['47575'], ['47126'], ['45851', '47147', '48451', '49748', '37668', '38966'], ['51950', '45867', '47164'], ['53249', '48460', '49756', '51080', '52378'], ['19453'], ['20185', '22175', '24187', '26189', '28185'], ['45863'], ['21633', '25633', '27622', '29618', '31617', '33603'], ['48030', '50024'], ['21817', '24690', '26466', '28243', '29769', '31479'], ['50307', '45704', '46998', '48294', '49600', '50900'], ['45923', '46823'], ['44713', '46565', '48418', '50271'], ['49704', '51558'], ['53178', '50622'], ['45416'], ['53046', '54092', '52413'], ['45057', '46919', '48775', '50628'], ['47303'], ['45262'], ['52747'], ['48634', '49931'], ['49215', '51290'], ['33010', '34832', '36654', '38476', '43022'], ['38946', '40803'], ['51260', '52554']]

# Set the directories used - UPDATE FOR YOUR OWN DIRECTORIES
directory = r'C:\Users\alice\Desktop\REU 2022\KOA_123815\HIRES\extracted\binaryfits\*\flux\*.fits.gz'

# gaia data file location - UPDATE TO YOUR OWN FILE LOCATION
gaia_file = r"C:\Users\alice\REU 2022\GaiaDR2_M15.fits.gz"
    
# In[Getting coordinates and coordinate matching with gaia]:
    
stars_v2, m15_coords = ss.get_coords(starnames)

gaia = fits.open(gaia_file)
gaia_data = gaia[1].data
  
gaia_coords = SkyCoord(gaia_data['ra'], gaia_data['dec'], unit=(u.deg, u.deg))
gaia_coords = gaia_coords.transform_to('icrs')    

mag_restrict = np.array(np.where((gaia_data['PHOT_G_MEAN_MAG'] < 18) & 
                        (gaia_data['PHOT_G_MEAN_MAG'] > 10) & 
                        np.isfinite(gaia_data['PHOT_BP_MEAN_MAG']) &
                        np.isfinite(gaia_data['PHOT_RP_MEAN_MAG'])))

mag_restrict = np.reshape(mag_restrict, (np.size(mag_restrict),1))

idx, d2d, d3d = m15_coords.match_to_catalog_sky(gaia_coords[mag_restrict].flatten())

mag_restrict = np.array(mag_restrict)

g_coords = gaia_coords[mag_restrict[idx]]

# name this stuff better?
gaia_IDs = gaia_data['source_id'][mag_restrict[idx]].flatten()
bp = gaia_data['PHOT_BP_MEAN_MAG'][mag_restrict[idx]].flatten()
rp = gaia_data['PHOT_RP_MEAN_MAG'][mag_restrict[idx]].flatten()
g = gaia_data['PHOT_G_MEAN_MAG'][mag_restrict[idx]].flatten()

pmra = gaia_data['pmra'][mag_restrict[idx]].flatten()
pmdec = gaia_data['pmdec'][mag_restrict[idx]].flatten()
pmra_err = gaia_data['pmra_error'][mag_restrict[idx]].flatten()
pmdec_err = gaia_data['pmdec_error'][mag_restrict[idx]].flatten()

# In[Setting up the HDU file]:
    
namecol = fits.Column(name ='name', array = stars_v2, format = '15A')
idcol = fits.Column(name = 'source_id', array = gaia_IDs, format = '15A')
racolm15 = fits.Column(name ='m15_ra', array = np.array(m15_coords.ra.deg), format = 'D')
deccolm15 = fits.Column(name ='m15_dec', array = np.array(m15_coords.dec.deg), format = 'D')
bpcol = fits.Column(name = 'bp_mag', array = bp, format = 'E')
rpcol = fits.Column(name = 'rp_mag', array = rp, format = 'E')
gcol = fits.Column(name = 'g_mag', array = g, format = 'E')

#columns to fill in
bp0col = fits.Column(name = 'bpmag0', array = np.zeros(len(stars_v2)), format = 'E')
rp0col = fits.Column(name = 'rpmag0', array = np.zeros(len(stars_v2)), format = 'E')
g0col = fits.Column(name = 'gmag0', array = np.zeros(len(stars_v2)), format = 'E')
teffcol = fits.Column(name = 'teff_mb20', array = np.zeros(len(stars_v2)), format = 'E')
loggcol = fits.Column(name = 'logg', array = np.zeros(len(stars_v2)), format = 'E')

coldefs = fits.ColDefs([namecol, idcol, racolm15, deccolm15, bpcol, rpcol, gcol,
                        bp0col, rp0col, g0col, teffcol, loggcol])

hdu = fits.BinTableHDU.from_columns(coldefs)

primary = fits.PrimaryHDU(np.array([1]))
hdul = fits.HDUList([primary, hdu])
hdul.writeto('gaia_m15_final.fits', overwrite = True)
hdul.close()

# In[Load in gaia_m15]:
    
hdul = fits.open('gaia_m15_final.fits')
gaia_m15 = hdul[1].data
hdul.close()

# In[Calculate mags, teff, logg...]:    
    
dm_m15 = 15.39
ebv_m15 = 0.10
feh_m15 = -2.37

# Gaia Collaboration, Babusiaux et al. 2018, A&A, 616, A10

kG = np.array([0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099])
kBP = np.array([1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043])
kRP = np.array([0.6104, -0.0170, -0.0026, -0.0017, -0.0078 ,0.00005, 0.0006])

# Mucciarelli & Bellazzini 2020, RNAAS

bprp_giant = np.array([0.5403, 0.4318, -0.0085, -0.0217, -0.0032, 0.0040])

# Initialize empty arrays to fill data into
gmag0 = np.array([])
bpmag0 = np.array([])
rpmag0 = np.array([])
teff_mb20 = np.array([])

#for i=0,ngaia_m15-1 do begin
for i in range(len(gaia_m15)):
    vars = [1.0, gaia_m15[i]['bp_mag'] - gaia_m15[i]['rp_mag'], 
            (gaia_m15[i]['bp_mag'] - gaia_m15[i]['rp_mag'])**2., 
            (gaia_m15[i]['bp_mag'] - gaia_m15[i]['rp_mag'])**3., 
            3.1*ebv_m15, (3.1*ebv_m15)**2., 
            (gaia_m15[i]['bp_mag'] - gaia_m15[i]['rp_mag'])*3.1*ebv_m15]
    gmag0 = np.append(gmag0, gaia_m15[i]['g_mag'] - sum(kG*vars)*3.1*ebv_m15)
    bpmag0 = np.append(bpmag0, gaia_m15[i]['bp_mag'] - sum(kBP*vars)*3.1*ebv_m15)
    rpmag0 = np.append(rpmag0, gaia_m15[i]['rp_mag'] - sum(kRP*vars)*3.1*ebv_m15)
    color = gaia_m15[i]['bpmag0'] - gaia_m15[i]['rpmag0']
    
    vars = [1e0, color, (color)**2., feh_m15, feh_m15**2., feh_m15*(color)]
    teff_mb20 = np.append(teff_mb20, 5040.0 / sum(bprp_giant*vars))
   
print(teff_mb20)
# Andrae et al. (2018, A&A, 616, A8)

sigma_SB = 5.6704e-5
G = 6.674e-8
Msun = 1.989e33
M = 0.75*Msun
teffdiff = teff_mb20 - 5772e0
bcG = np.polyval([-7.197e-15, 2.859e-11, -6.647e-8, 6.731e-5, 6e-2], teffdiff)
logL = (gmag0 - (dm_m15 - 2.682*ebv_m15) + bcG - 4.74) / (-2.5e0) + np.log10(3.828e33)
logg_arr = np.log10(4 * np.pi * sigma_SB * G) + np.log10(M) + 4.*np.log10(teff_mb20) - logL

# In[Add gmag0, bpmag0, rpmag0, and teff_mb20 to the data table]:

hdul = fits.open('gaia_m15_final.fits', mode='update')

hires = hdul[1].data
hires['gmag0'] = gmag0
hires['bpmag0'] = bpmag0
hires['rpmag0'] = rpmag0
hires['teff_mb20'] = teff_mb20
hires['logg'] = logg_arr

hdul.flush()
hdul.close()

# In[Change star names to not have slash or space]:
    
hdul = fits.open('gaia_m15_final.fits', mode='update')

data = hdul[1].data
for i in range(len(data['name'])):
    star = data['name'][i]
    star=star.replace('/','_')
    star=star.replace('\\','_')
    star=star.replace(' ','_')
    star=star.replace(',','_')   
    star=star.replace('_(Flux)','') 

hdul.flush()
hdul.close()

# In[Since two M15_K969's happen...]:
    
hdul = fits.open('gaia_m15_final.fits', mode='update')

data = hdul[1].data
data['name'][25]='M15-K969'   

hdul.flush()
hdul.close()

