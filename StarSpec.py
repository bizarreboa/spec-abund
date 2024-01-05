#!/usr/bin/env python
# coding: utf-8

# For a list of names of stars, like the list 'stars' defined in the initialization cell,
# you'll just want to run the function write_data(stars). It will make you sit there and
# wait for a long time, every now and then asking you for the redshifted wavelength, so
# you might want to break it up into chunks.

# In[Initialization]:

import astropy.nddata as nddata
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import re
from specutils import Spectrum1D
from specutils.manipulation import LinearInterpolatedResampler, gaussian_smooth
linr = LinearInterpolatedResampler()
import astropy.units as u
from specutils.fitting import fit_generic_continuum
from astropy.coordinates import SkyCoord

# Global plot formatting
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

mpl.rcParams['agg.path.chunksize'] = 1000000

# List of exposure IDs from the database
exps = [['51990'], ['54353'], ['50889', '51435', '51981', '51347','48271', '50181', '52090', 
                               '21817', '24690', '26466', '28243', '29769', '31479'], 
        ['39707', '41219', '47488', '39833'], 
        ['36335', '38324', '40315', '42303', '44614', '46610', '48599'], 
        ['32465', '36476', '38471', '40465', '43119', '45186', '47180'], 
        ['40897', '42463', '44030', '45596'], ['34706'], ['49605', '50700'], 
        ['29313', '31183', '33277', '35138', '36994'], ['51794', '51664'], ['47575'], ['47126'], 
        ['45851', '47147', '48451', '49748', '37668', '38966'], ['51950', '45867', '47164'], 
        ['53249', '48460', '49756', '51080', '52378'], ['19453'], ['20185', '22175', '24187', '26189', '28185'], 
        ['45863'], ['21633', '25633', '27622', '29618', '31617', '33603'], ['48030', '50024'], 
        ['50307', '45704', '46998', '48294', '49600', '50900'], ['45923', '46823'], 
        ['44713', '46565', '48418', '50271'], ['49704', '51558'], ['53178', '50622'], ['45416'], 
        ['53046', '54092', '52413', '33010', '34832', '36654', '38476', '43022'], 
        ['45057', '46919', '48775', '50628'], ['47303'], ['45262'], ['52747'], ['48634', '49931'], 
        ['49215', '51290'], ['38946', '40803'], ['51260', '52554']]

# Old list of stars downloaded from the database
starnames = ['M15C29294_1631', 'HD204712', 'M15_K969', 'H204712', 
            'K386', 'K462', 'M15_K490', 'M15-C29523_1114', 
            'M15C31264_1605', 'm15c29413x1023', 'M15_I-63', 'm15bcf315', 
            'M15_III-59', 'C30131_0829', 'K731_I-63', 'K934_I-62', 
            'K479', 'K341', 'm15bcf203', 'K583', 'II-64', 'C30247_0911', 
            'M15C30599_2226', 'M15-G30482_1054', 'M15-pr109', 
            'M15C30027_0019', 'M56_I_22', 'M15_K431', 
            'M15C29589_1223', 'K582_IV-62', 'M15-K853', 'M15_I-62', 
            '1-38', 's3', 'M15_IV-62', 'K1084']

# Updated list of more usable / actually in M15 stars
stars = ['M15_K969','K386','K462','k969_m15','M15_III-59', 'K479', 'K341', 'K583',
         'II-64', 'M15_K431','M15-K853', 's3', 'M15C29294_1631', 'HD204712', 'H204712',
         'M15_K490', 'M15-C29523_1114', 'M15C31264_1605', 'm15c29413x1023', 'M15_I-63',
         'm15bcf315', 'C30131_0829', 'K731_I-63', 'K934_I-62', 'm15bcf203',
         'C30247_0911', 'M15C30599_2226', 'M15-G30482_1054', 'M15-pr109',
         'M15C30027_0019', 'M56_I_22', 'M15C29589_1223', 'M15_I-62',
         '1-38', 'M15_IV-62', 'K1084']

# Create range of evenly-spaced wavelengths and flux to be used later
wave_arr = np.arange(3150,9100.01,0.01)
f_wave = np.arange(3000,8000,0.02)
f_flux = np.zeros(len(f_wave))
f_wts = np.copy(f_flux)

# Initialize some constants
ba = 4554.034
other_ba = 6141.730 #4934.100
fe = 5098.6697 #5110.413
ni = 3807.140

c = 2.998e5 # In km/s

# Initiliaze delimiters
delims = r"[._]"

# Set the directories used - UPDATE FOR YOUR OWN DIRECTORIES
directory = r'C:\Users\alice\Desktop\REU 2022\KOA_123815\HIRES\extracted\binaryfits\*\flux\*.fits.gz'
out2dir = r'C:/Users/alice/REU 2022/out2_models/'

# In[Main necessary functions]:
    
def get_starnames(directory):
    """
    From the given directory, returns the names of all 
    the stars included as a list.
    """
    
    names = []
    # Get the names of all the observed stars
    for filename in glob.iglob(f'{directory}'):
        hdul = fits.open(filename)
        heads = hdul[0].header
        if 'TARGNAME' in heads:
            names.append(heads['TARGNAME'])
        elif 'OBJECT' in heads:
            names.append(heads['OBJECT'])
    return set(names)

def get_star_exps(targname):
    """
    Collects all the exposures for a given star into a
    list.
    
    Args
    ---------------------------------------------------
        targname:   the name of the star/object
    
    Returns
    ---------------------------------------------------
        exposures:  a list of exposure IDs associated with 
                    the given targname
    """
    # Initialize empty list to hold data
    exposures = []

    # Search through the directory
    for filename in glob.iglob(f'{directory}'):
        hdul = fits.open(filename)
        heads = hdul[0].header
        
        # Get the name of the star
        if 'TARGNAME' in heads:
            targ = heads['TARGNAME']
            if targ == str(targname):
                # Add the exposure ID to the list
                expID = re.split(r"[._]", filename.rsplit('\\', 1)[-1])[2]
                exposures.append(expID)                
        else:
            # If not in TARGNAME, it will be in OBJECT
            targ = heads['OBJECT']
            if targ == str(targname):
                # Add the exposure ID to the list
                expID = re.split(r"[._]", filename.rsplit('\\', 1)[-1])[2]
                exposures.append(expID)
    
    # Get unique exposures numbers only
    exposures = set(exposures)
    
    return exposures 

def get_exposure_orders(expID):
    """
    Returns a list of all the filenames with a given
    exposure ID.
    """
    res = []
    for filename in glob.iglob(f'{directory}'):
        file = filename.rsplit('\\', 1)[-1]
        exp = re.split(delims, file)[2]

        if exp == str(expID):
            res.append(str(filename))

    return res

def cont_norm(spec):
    """
    Continuum-normalizes the given spectrum object, spec.
    Returns the normalized version of the spectrum.
    """
    flux = spec.flux.value.flatten()
    wave = spec.spectral_axis.flatten().value
    wts = spec.uncertainty.quantity.flatten()
    
    m = np.mean(flux)
    s = np.std(flux)
    c_p = zip(wave, flux, wts)
    c_p = [x for x in c_p if x[1] > m-s and np.isfinite(x[1]) and 
         np.isfinite(x[2]) and x[1] < m+2*s]
    c_trimwave, c_trimflux, c_trimweights = zip(*c_p)

    #err = [x**(-0.5) for x in c_trimweights]
    spec = make_spec_obj(c_trimwave, c_trimflux, c_trimweights)
    #continuum = gaussian_smooth(spec1, stddev=3)
    continuum = fit_generic_continuum(spec)
    #f = interp.UnivariateSpline(c_trimwave, c_trimflux, w=c_trimweights, k=3, s=500)
    #continuum = f(wave)

    #tck = interp.splrep(c_trimwave, c_trimflux, w=c_trimweights, k=3, s=1000, task=0)
    #continuum = interp.splev(wave, tck, der=0)

    error = [x**(-0.5) for x in wts]
    contdiv = flux / continuum(wave*u.Angstrom)
    errordiv = error / continuum(wave*u.Angstrom)
    wtsdiv = errordiv**(-2)
    
    return make_spec_obj(wave, contdiv, wtsdiv)

def plot_continuum(spec):
    """
    Plots the continuum of a given spectrum as
    a Spec1D object, spec
    """
    
    flux = spec.flux.value.flatten()
    wave = spec.spectral_axis.flatten().value
    wts = spec.uncertainty.quantity.flatten()
    
    m = np.mean(flux)
    s = np.std(flux)
    c_p = zip(wave, flux, wts)
    c_p = [x for x in c_p if x[1] > m-2*s and np.isfinite(x[1]) and 
         np.isfinite(x[2]) and x[1] < m+2*s]
    c_trimwave, c_trimflux, c_trimweights = zip(*c_p)
    c_trimwave, c_trimflux, c_trimweights = np.array(c_trimwave), np.array(c_trimflux), np.array(c_trimweights)

    spec = make_spec_obj(wave, flux, wts)
    #continuum = gaussian_smooth(spec1, stddev=3)
    continuum = fit_generic_continuum(spec)
    #f = interp.UnivariateSpline(c_trimwave, c_trimflux, w=c_trimweights, k=3, s=500)
    #continuum = f(wave)

    #tck = interp.splrep(c_trimwave, c_trimflux, w=c_trimweights, k=3, s=1000, task=0)
    #continuum = interp.splev(wave, tck, der=0)
    
    plt.plot(wave, flux)
    plt.plot(wave, continuum(wave*u.Angstrom))
    plt.show()

def make_spec_obj(wave, flux, wts):
    """
    Returns given spectrum data as a spectrum object
    
    Args
    ---------------------------------------------------
        wave:   list or array of wavelengths
        flux:   list or array of flux
        wts:    list or array of weights for each flux
        
    Returns
    ---------------------------------------------------
        specobj: Spectrum1D object
    """

    # Make sure we have numpy arrays
    wave, flux, wts = np.array(wave), np.array(flux), np.array(wts)
    
    specobj = Spectrum1D(flux = flux*u.dimensionless_unscaled, 
                         spectral_axis = wave*u.Angstrom, 
                         uncertainty = nddata.InverseVariance(wts))
    return specobj

def coadd(spec, f_spec):
    """
    Expects Spectrum1D objects.
    
    Rebins spec and coadds the rebinned spectrum 
    onto f_spec, the final spectrum.
    
    Returns resampled, coadded Spectrum1D object.
    """
    # Get the wavelength data from the Spectrum1D objects
    #wave, f_wave = spec.spectral_axis, f_spec.spectral_axis
    
    # Rebin the spectrum to match f_spec
    rb_spec = linr(spec, f_spec.spectral_axis.flatten().quantity)
    
    # Get the data from the Spectrum1D objects
    flux, wts = rb_spec.flux, rb_spec.uncertainty.quantity.flatten()
    f_flux, f_wts = f_spec.flux, f_spec.uncertainty.quantity.flatten() 
    
    flux = np.nan_to_num(flux)
    f_flux = np.nan_to_num(f_flux)
    wts = np.nan_to_num(wts, nan=10**(-12))
    f_wts = np.nan_to_num(f_wts, nan=10**(-12))
    
    # Get the weighted average for the flux
    new_flux = np.nan_to_num(np.average(np.array([flux, f_flux]), axis = 0, 
                     weights = np.array([wts, f_wts])).flatten())

    # Get the sum of the weights for each bin
    new_wts = np.nan_to_num(np.sum(np.array([wts, f_wts]), axis = 0).flatten())

    new_wave = f_spec.spectral_axis.flatten().value
    
    return make_spec_obj(new_wave, new_flux, new_wts)
    #return linr(new_spec, f_spec.spectral_axis.flatten().quantity)

def get_synth_spec(starname):
    """
    Opens the .out2 file for the star and reads in the 
    flux values, returning a Spectrum1D object containing 
    the flux and wave array.
    
    Args
    ---------------------------------------------------
        starname:   name of the star as a string
    
    Returns
    ---------------------------------------------------
        synth_spec: the synthetic model spectrum of the star
                    as a spectrum1D object
    """
    file = out2dir+starname+'.out2'
    with open(file) as f:
        lines = f.read()
    new_lines = lines.replace('-',' -')
    str_values = np.array(list(filter(None,re.split(r"\n| ",new_lines))))[36:]
    values = np.asarray(str_values, dtype = float)
    values = 1-values
    synth_spec = Spectrum1D(flux = values*u.dimensionless_unscaled, 
                            spectral_axis = wave_arr*u.Angstrom)
    return synth_spec

def cont_norm_model(spec, synth_spec):
    """
    Continuum-normalizes the given spectrum object.
    Returns the normalized version of the spectrum.
    
    Args
    ---------------------------------------------------
        spec:       spectrum1D object of the spectrum to be
                    normalized
        synth_spec: spectrum1D object of the syntheized
                    model spectrum
    
    Returns
    ---------------------------------------------------
        the normalized spectrum using model spectrum, 
        as a spectrum1D object
    """
    flux = spec.flux.value.flatten()
    wave = spec.spectral_axis.flatten().value
    wts = spec.uncertainty.quantity.flatten()
    
    m = np.mean(flux)
    s = np.std(flux)
    p = zip(wave, flux, wts)
    p = [x for x in p if x[1] > m-2*s and np.isfinite(x[1]) and 
         np.isfinite(x[2]) and x[1] < m+2*s]
    trimwave, trimflux, trimweights = zip(*p)

    center_lambda = (np.amin(wave) + np.amax(wave))/2
    sigma = (center_lambda/30000.)/2.35
    thic_synth_spec = gaussian_smooth(synth_spec, stddev = sigma/0.01)

    rb_synth_spec = linr(thic_synth_spec, np.array(trimwave)*u.Angstrom)
    synth_flux = rb_synth_spec.flux.value.flatten()
    
    divflux = trimflux / synth_flux
    error = [x**(-0.5) for x in trimweights]
    errordiv = error / synth_flux
    divwts = errordiv**(-2)
    
    divspec = make_spec_obj(np.array(trimwave), divflux, divwts)    
    continuum = fit_generic_continuum(divspec)
    
    err = [x**(-0.5) for x in wts]
    contdiv = flux / continuum(wave*u.Angstrom)
    errdiv = err / continuum(wave*u.Angstrom)
    wtsdiv = errdiv**(-2)
    
    return make_spec_obj(wave, contdiv, wtsdiv)

def get_exp_spec(orders, synth_spec, redshift):
    """
    Collects all the data for a given exposure ID.
    Stitches orders together using the coadd function.
    
    Args
    ---------------------------------------------------
        expID:      the exposure ID
    
    Returns
    ---------------------------------------------------
        exp_spec:   Spectrum1D object for the stitched exposure
    """
    # Set up the full exposure spectrum
    f_wave = np.arange(3000,8000,0.02)
    f_flux = np.zeros(len(f_wave))
    f_wts = np.copy(f_flux)
    exp_spec = make_spec_obj(f_wave, f_flux, f_wts)
    
    for file in orders:
        hdul = fits.open(file)
        hires = hdul[1].data
        # de-redshift the wave when making the spectrum object
        spec = make_spec_obj(hires['wave']/(1 + redshift), hires['flux'], 
                             hires['error']**(-2))
        order_spec = cont_norm_model(spec, synth_spec)
        exp_spec = coadd(order_spec, exp_spec)
        
    plt.show()
    
    return exp_spec
    
def ask_for_wavelength():
    """
    Asks for user input of the observed wavelength
    of the line being plotted.
    """
    measured_wave = float(input("Observed wavelength is: "))
    return measured_wave

def get_redshift(orders):
    """
    Given the list of orders of an exposure, finds the 
    order with a strong absorption line.
    Plots this order and asks for the observed wavelength,
    then calculates and returns the redshift.
    """
    notinorder = True
    i = 0
    
    hdul0 = fits.open(orders[0])
    hires0 = hdul0[1].data
    hdul0.close()
    
    hdulf = fits.open(orders[-1])
    hiresf = hdulf[1].data
    hdulf.close()
    
    # if the first order starts after the BaII 4554 line, 
    # find the BaII 6142 line and use that
    if (np.amin(hires0['wave']) > ba):
        while notinorder:
            hdul = fits.open(orders[i])
            hires = hdul[1].data
            notinorder = (np.min(hires['wave']) > other_ba) or (np.max(hires['wave']) < other_ba)
            i+=1
            hdul.close()
        plot_around(hires, other_ba, 'BaII 6142')
        measured_wave = ask_for_wavelength()
        redshift = (measured_wave - other_ba)/other_ba
        plt.close()
        
    # if the last order ends before the BaII 4554 line,
    # find the NiI 3807 line and use that
    elif (np.amax(hiresf['wave']) < ba):
        while notinorder:
            hdul = fits.open(orders[i])
            hires = hdul[1].data
            notinorder = (np.min(hires['wave']) > ni) or (np.max(hires['wave']) < ni)
            i+=1
            hdul.close()
        plot_around(hires, ni, 'NiI 3807')
        measured_wave = ask_for_wavelength()
        redshift = (measured_wave - ni)/ni
        plt.close()
        
    # otherwise, use the BaII 4554 line
    else:
        while notinorder:
            hdul = fits.open(orders[i])
            hires = hdul[1].data
            notinorder = (np.min(hires['wave']) > ba) or (np.max(hires['wave']) < ba)
            i+=1
            hdul.close()
        plot_BaII_order(hires)
        measured_wave = ask_for_wavelength()
        redshift = (measured_wave - ba)/ba
        plt.close()
    
    return redshift
  
def plot_around(hires, wavelength, linename):
    """
    Plots the hires data from an order, specifically around
    some specified wavelength of linename.
    """
    wave = hires['wave']
    flux = hires['flux']
    
    plt.figure(figsize=(9,6))
    plt.plot(wave, flux)
    plt.title(str(linename) + ' line')
    plt.xlabel('wavelength (Å)')
    plt.ylabel('continuum-normalized flux')
    plt.xlim(wavelength-4, wavelength+4)
    plt.xticks(np.arange(wavelength-2.5, wavelength+1, 0.05), rotation=45)
    plt.draw()
    plt.show(block=False)
    

def plot_BaII_order(hires):
    """
    Given the order data in hires, plots the region near 
    the BaII 4554 absorption line to help figure out the 
    redshift.
    """    
    wave = hires['wave']
    flux = hires['flux']
    
    plt.figure(figsize=(9,6))
    plt.plot(wave, flux)
    plt.title('BaII 4554 line')
    plt.xlabel('wavelength (Å)')
    plt.ylabel('continuum-normalized flux')
    plt.xlim(4550, 4558)
    plt.xticks(np.arange(4550.5, 4555, 0.05), rotation=45)
    plt.draw()
    plt.show(block=False)

def plot_other_order(hires):
    """
    Given the order data in hires, plots the region near 
    the other absorption line to help figure out the 
    redshift.
    """    
    wave = hires['wave']
    flux = hires['flux']
    
    plt.figure(figsize=(9,6))
    plt.plot(wave, flux)
    plt.title('BaII 4934 line')
    plt.xlabel('wavelength (Å)')
    plt.ylabel('continuum-normalized flux')
    plt.xlim(4930, 4938)
    plt.xticks(np.arange(4932.5, 4935, 0.05), rotation=45)
    plt.draw()
    plt.show(block=False)

def get_full_spec(starname):
    """
    Given the name of the star, returns the fully stitched, 
    stacked, and coadded data as a Spectrum1D object.
    """
    # Get the list of exposures for this star
    exposures = exps[starnames.index(starname)]
    print("Number of exposures:",len(exposures))
    
    # Get the synthetic model spectrum for the star
    synth_spec = get_synth_spec(starname)
    
    # Get the filenames for one exposure to check BaII
    orders = get_exposure_orders(exposures[-1])
    
    # Find the order that has the BaII line, leaving the hires data in memory
    redshift = get_redshift(orders)
    
    # Set up the final spectrum for this star
    f_wave = np.arange(3000,8000,0.02)
    f_flux = np.zeros(len(f_wave))
    f_wts = np.copy(f_flux)
    f_spec = make_spec_obj(f_wave, f_flux, f_wts)
    
    if len(exposures) == 0:
        print("No exposures???")
        
    else:
        for exp in exposures:
            exp_spec = get_exp_spec(orders, synth_spec, redshift)
            f_spec = coadd(exp_spec, f_spec)
            
        wave = f_spec.spectral_axis.flatten().value
        flux = f_spec.flux.flatten()
        wts = f_spec.uncertainty.quantity.flatten()
        
        nonzero = np.argwhere(flux != 0)
        
        return make_spec_obj(wave[nonzero].flatten(), flux[nonzero].flatten(), 
                             wts[nonzero].flatten()), redshift
    
def find_mjd(starname):
    """
    Finds the first file found for a given star,
    and returns the mjd of that file
    """
    
    # Search through the directory
    for filename in glob.iglob(f'{directory}'):
        hdul = fits.open(filename)
        heads = hdul[0].header
        
        # Get the name of the star
        if 'TARGNAME' in heads:
            targ = heads['TARGNAME']
            if targ == str(starname):
                return float(heads['mjd'])  
        else:
            # If not in TARGNAME, it will be in OBJECT
            targ = heads['OBJECT']
            if targ == str(starname):
                return float(heads['mjd'])
                
        hdul.close()
    
def add_vr2file(star, redshift):
    """
    Given a star and the reshift found for it,
    updates the .fits file of the star with
    this redshift value.
    """
    
    f = fits.open(star+'.fits', mode='update')
    
    hires = f[1].data
    hires['vr_real'] = np.array([redshift*c])
    
    f.flush()
    f.close()
 
def write_spec(star, spec, redshift):
    """
    Writes a spectrum1D object for star with redshift
    to a new .fits file.
    """

    flux = spec.flux.value.flatten()
    wave = spec.spectral_axis.flatten().value
    wts = spec.uncertainty.quantity.flatten()
    
    mjd = find_mjd(star)
    vr = c*redshift
    
    length = str(len(wave))
        
    wavecol = fits.Column(name='lambda', array=np.array(wave).reshape(1,len(wave)), format = length+'D')
    fluxcol = fits.Column(name='spec', array=np.array(flux).reshape(1,len(wave)), format = length+'D')
    wtscol = fits.Column(name='ivar', array=np.array(wts).reshape(1,len(wave)), format = length+'D')
    mjdcol = fits.Column(name='mjdavg', array=np.array([mjd]), format = 'D')
    vrcol = fits.Column(name='vr', array=np.array([0]), format = 'D')
    vr2col = fits.Column(name='vr_real', array=np.array([vr]), format = 'D')

    coldefs = fits.ColDefs([wavecol, fluxcol, wtscol, mjdcol, vrcol, vr2col])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    primary = fits.PrimaryHDU(np.array([1]))
    hdul = fits.HDUList([primary, hdu])
    hdul.writeto(star+'.fits', overwrite = True)

    hdul.close()

def write_data(stars):
    """
    For a list of stars, writes in the data
    for all of them into new .fits files.
    
    This is essentially the main function that will
    write in the proper spectrum for your stars.
    """
    for star in stars:
        spec, redshift = get_full_spec(star)
        write_spec(star, spec, redshift)

# In[Functions to get coordinates for coordinate-matching]:
    
def get_exps_coords(star):
    """
    Get the coordinates of star.
    """
    
    exposures = []
    ras_fk5 = []
    decs_fk5 = []
    ras_fk4 = []
    decs_fk4 = []

    # Search through the directory
    for filename in glob.iglob(f'{directory}'):
        hdul = fits.open(filename)
        heads = hdul[0].header
        
        # Get the name of the star
        if 'TARGNAME' in heads:
            targ = heads['TARGNAME']
            if targ == str(star):
                # Add the exposure ID to the list
                expID = re.split(r"[._]", filename.rsplit('\\', 1)[-1])[2]
                if expID not in exposures:
                    exposures.append(expID)
                    
                    if heads['frame'] == 'FK4':
                        ras_fk4.append(heads['ra'])
                        decs_fk4.append(heads['dec'])
                        
                    elif heads['frame'] == 'FK5':
                        ras_fk5.append(heads['ra'])
                        decs_fk5.append(heads['dec'])
        else:
            # If not in TARGNAME, it will be in OBJECT
            targ = heads['OBJECT']
            if targ == str(star):
                # Add the exposure ID to the list
                expID = re.split(r"[._]", filename.rsplit('\\', 1)[-1])[2]
                if expID not in exposures:
                    exposures.append(expID)     
                    
                    if heads['frame'] == 'FK4':
                        ras_fk4.append(heads['ra'])
                        decs_fk4.append(heads['dec'])
                        
                    elif heads['frame'] == 'FK5':
                        ras_fk5.append(heads['ra'])
                        decs_fk5.append(heads['dec'])
                    
    return exposures, ras_fk4[0], decs_fk4[0], ras_fk5[0], decs_fk5[0]

def get_one_file(star):
    """
    Returns the first file found for a given star.
    """
    for filename in glob.iglob(f'{directory}'):
        hdul = fits.open(filename)
        heads = hdul[0].header
        
        # Get the name of the star
        if 'TARGNAME' in heads:
            targ = heads['TARGNAME']
            if targ == str(star):
                return filename
        else:
            # If not in TARGNAME, it will be in OBJECT
            targ = heads['OBJECT']
            if targ == str(star):
                return filename
            
def get_coords(stars):
    """
    Returns the list of stars in the same order as
    the coordinates that are returned in a 
    SkyCoord object. 
    
    Returns the SkyCoord object containing all 
    the coordinates in degrees.
    """
    stars_fk5 = []
    ras_fk5 = []
    decs_fk5 = []
    stars_fk4 = []
    ras_fk4 = []
    decs_fk4 = []
    
    for star in stars:
        file = get_one_file(star)
        hdul = fits.open(file)
        heads = hdul[0].header
        
        if heads['frame'] == 'FK4':
            stars_fk4.append(star)
            ras_fk4.append(heads['ra'])
            decs_fk4.append(heads['dec'])
            
        elif heads['frame'] == 'FK5':
            stars_fk5.append(star)
            ras_fk5.append(heads['ra'])
            decs_fk5.append(heads['dec'])
    
    coords_fk4 = SkyCoord(ras_fk4, decs_fk4, unit=(u.hourangle, u.deg), frame='fk4')
    coords_fk5 = SkyCoord(ras_fk5, decs_fk5, unit=(u.hourangle, u.deg), frame='fk5')
    
    new_coords_4 = coords_fk4.transform_to('icrs')
    new_coords_5 = coords_fk5.transform_to('icrs')
    
    ras = list(new_coords_4.ra.deg) + list(new_coords_5.ra.deg)
    decs = list(new_coords_4.dec.deg) + list(new_coords_5.dec.deg)
    stars_reordered = list(stars_fk4) + list(stars_fk5)
    
    return stars_reordered, SkyCoord(ras, decs, unit=(u.deg, u.deg), frame = 'icrs')

# In[Function to test the continuum model]:
    
def test_cont(star, spec):
    """
    Given the starname and its spectrum data,
    plots the model, continuum and data.
    """
    exposures = exps[starnames.index(star)]
    print("Number of exposures:",len(exposures))
    
    # Get the synthetic model spectrum for the star
    synth_spec = get_synth_spec(star)
    
    # Get the filenames for one exposur
    orders = get_exposure_orders(exposures[-1])
    
    # Find the order that has a strong absorption line
    # and return the redshift
    redshift = get_redshift(orders)
    
    # Set up the final spectrum for this star
    f_wave = np.arange(3000,8000,0.02)
    f_flux = np.zeros(len(f_wave))
    f_wts = np.copy(f_flux)
    f_spec = make_spec_obj(f_wave, f_flux, f_wts)
    
    if len(exposures) == 0:
        print("No exposures???")
        
    else:
        for exp in exposures:
            exp_spec = get_exp_spec(orders, synth_spec, redshift)
            f_spec = coadd(exp_spec, f_spec)
            
        wave = f_spec.spectral_axis.flatten().value
        flux = f_spec.flux.flatten()
        wts = f_spec.uncertainty.quantity.flatten()
        
        nonzero = np.argwhere(flux != 0)
        
        return make_spec_obj(wave[nonzero].flatten(), flux[nonzero].flatten(), 
                             wts[nonzero].flatten()), redshift
    
# In[Other plotting functions just to look at stuff]:

def plotting(starname, spec):
    """
    General plotting function to plot the spectrum of
    a star called starname, just to take a look.
    """
    
    wave = spec.spectral_axis.flatten()
    flux = spec.flux.flatten()
    wts = spec.uncertainty.quantity.flatten()
    
    print(f"""_____________________________________
Star: {starname}
***********************************
Min wavelength: {np.amin(wave)}
Max wavelength: {np.amax(wave)}

Min flux: {np.amin(flux)}
Max flux: {np.amax(flux)}

Min weight: {np.amin(wts)}
Max weight: {np.amax(wts)}""")
    
    plt.figure(figsize=(30,10))
    plt.plot(wave, flux)
    plt.xlabel('wavelength (Å)')
    plt.ylabel('continuum-normalized flux')
    plt.title(starname)
    plt.show()

def plot_star(starname):
    """
    Given starname, plots the full coadded spectrum.
    Also prints max/min values.
    """
    full_spec = get_full_spec(starname)
    plotting(starname, full_spec)
    
def plot_order(order_file, expID):
    """
    Given a file, plots the order.
    """
    hdul = fits.open(order_file)
    hires = hdul[1].data
    
    wave = hires['wave']
    flux = hires['flux']
    wts = hires['error']**(-2)
    
    spec = make_spec_obj(wave, flux, wts)
    
    print(f"""
***********************************
Min wavelength: {np.amin(wave)}
Max wavelength: {np.amax(wave)}

Min flux: {np.amin(flux)}
Max flux: {np.amax(flux)}

Min weight: {np.amin(wts)}
Max weight: {np.amax(wts)}""")
    
    rb_spec = linr(spec, np.arange(3000,8000,0.02)*u.Angstrom)
    
    plt.figure(figsize=(30,10))
    plt.plot(rb_spec.spectral_axis.flatten(), rb_spec.flux.flatten())
    plt.xlabel('wavelength (Å)')
    plt.ylabel('flux')
    plt.title(expID)
    plt.show()
    
def plot_spec(spec1D):
    """
    Plots a given a spectrum1D object.
    """
    wave = spec1D.spectral_axis.flatten()
    flux = spec1D.flux.flatten()
    
    plt.figure(figsize=(30,10))
    plt.plot(wave, flux)
    plt.xlabel('wavelength (Å)')
    plt.ylabel('flux')
    plt.show()
    
    