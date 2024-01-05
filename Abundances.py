# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:23:00 2022

@author: alice
"""
# You'll mainly want to use the two functions at the end, main and plot_mgeu
# main will guide you through the process of getting abundances and
# hyperfine abundances for the stars that have been measured already.
# plot_mgeu will give you the graph for mg vs eu abundances for these stars

# In[Initialization]:
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from specutils.manipulation import LinearInterpolatedResampler
linr = LinearInterpolatedResampler()

# Global plot formatting
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

el_sym = ['H', 'He','Li','Be','B', 'C', 'N', 'O', 'F', 'Ne', 
        'Na','Mg','Al','Si','P', 'S', 'Cl','Ar','K', 'Ca', 
        'Sc','Ti','V', 'Cr','Mn','Fe','Co','Ni','Cu','Zn',
        'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y', 'Zr', 
        'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', 
        'Sb','Te','I', 'Xe','Cs','Ba','La','Ce','Pr','Nd',
        'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', 
        'Lu','Hf','Ta','W', 'Re','Os','Ir','Pt','Au','Hg', 
        'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', 
        'Pa','U', 'Np','Pu','Am']

solar_ab = [12.00,10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93, 
             6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,  
             3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56,  
             3.04, 3.65, 2.30, 3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58, 
             1.46, 1.88,-5.00, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,  
             1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58, 0.72, 1.42,  
             -5.00, 0.96, 0.52, 1.07, 0.30, 1.10, 0.48, 0.92, 0.10, 0.84,  
             0.10, 0.85,-0.12, 0.85, 0.26, 1.40, 1.38, 1.62, 0.92, 1.17,  
             0.90, 1.75, 0.65,-5.00,-5.00,-5.00,-5.00,-5.00,-5.00, 0.02,  
             -5.00,-0.54,-5.00,-5.00,-5.00]

hf_elems = [21, 23, 25, 27, 29, 56, 57, 60, 63]

general_hf_file = 'C:/Users/alice/REU 2022/hyperfine.par'

# Read in the hyperfine linelist dataframe
# UPDATE THE DIRECTORY TO THE FILE TO YOUR OWN!!!
hf_file = 'C:/Users/alice/REU 2022/blend_examples/Ji20_hyperfine.moog'
hfdf = pd.read_csv(hf_file, delim_whitespace = True, 
                   names = ['wave', 'iso','col3', 'col4'])
hfdf['empty1'] = pd.Series(dtype = 'float')
hfdf['empty2'] = pd.Series(dtype = 'float')
hfdf['ewcol'] = pd.Series(dtype = 'float')

iso_tab = pd.read_fwf('C:/Users/alice/REU 2022/tab3.txt', widths = [8,8,9]).ffill()
r_iso_tab = pd.read_fwf('C:/Users/alice/REU 2022/tab1.txt').ffill()

mgeustars = ['M15C30599_2226','K1084','M15-C29523_1114','M15-G30482_1054',
             'M15C29589_1223','M15C30027_0019','M15_I-62','M15_I-63',
             'M15_IV-62','M15_K431','M15_K490','M15_K969']

# Set the directories used - UPDATE FOR YOUR OWN DIRECTORIES
out2abs_dir = 'C:/Users/alice/REU 2022/out2_abunds/'
otherpars_dir = 'C:/Users/alice/REU 2022/otherpars/'
ews_dir = 'C:/Users/alice/REU 2022/ews/'
hfews_dir = 'C:/Users/alice/REU 2022/hfews/'
pars_dir = 'C:/Users/alice/REU 2022/pars/'
out2hfabs_dir = 'C:/Users/alice/REU 2022/out2_hfabunds/'
star_nat_file = 'C:/Users/alice/REU 2022/star_nat.par'
star_me_file = 'C:/Users/alice/REU 2022/star_me.par'

# In[from pymoogi, https://github.com/madamow/pymoogi/blob/master/pymoogi/lib/read_out_files.py]:
    
def out2_abfind(file):
    """
    Given a .out2 file, converts the data into
    a pandas dataframe.
    """
    
    # read filename from commandline
    with open(file, 'r') as fh:
        data = fh.readlines()

    line_data = np.zeros((1, 8), dtype=float)

    for line in data:
        line = re.split('\s+', line.strip())

        try:
            line = np.array(line, dtype=float)
            line_data = np.append(line_data, np.array([line]), axis=0)
        except:
            pass

    # turn data into pandas dataframe with headers
    df = pd.DataFrame(data = line_data[1:, :], 
                      columns=['wave', 'elem', 'ep', 'logGF', 'EWin', 
                               'logRWin', 'abund', 'delavg'])
    
    # for hyperfine blend out2 files, cull the 999.990 lines
    if 999.990 in df.abund.values:
        df = df.drop(df[(df.abund > 900.)].index)
    
    return df
  
def met_XXsun(X, abX, stdX):
    """
    Gives the abundance ratio [X/Xsun]
    
    Args
    ---------------------------------------------------
        X:      element number (or symbol, perhaps?)
        abX:    abundance of X
        stdX:   standard deviation of abX
    
    Returns
    ---------------------------------------------------
        met:    The abundance ratio [X/Xsun]
        std:    error in met
    """
    ab_sun = solar_abund(X)
    met = abX - ab_sun
    return met, stdX

def met_XY(df, X, Y):
    """
    Gives the abundance ratio for [X/Y].
    
    Args
    ---------------------------------------------------
        df: star out2 table dataframe
        X:  element number of first element
        Y:  element number of second element
        
    Returns
    ---------------------------------------------------
        met: The abundance ratio [X/Y]
        std: error in met
    """
    abX, stdX = get_elem_data(X, df)
    abY, stdY = get_elem_data(Y, df)
    
    abxh, stdxh = met_XXsun(X, abX, stdX)
    abyh, stdyh = met_XXsun(Y, abY, stdY)
    
    met = abxh - abyh
    std = np.sqrt(stdxh**2 + stdyh**2)
    
    return met, std

# make met_XH callable by itself (median abX), to get X/H
def met_XH(df, X):
    """
    Gives the abundance ratio for [X/Hsun].
    
    Args
    ---------------------------------------------------
        df:  star out2 table dataframe
        X:   element number
        
    Returns
    ---------------------------------------------------
        met: The abundance ratio [X/Y]
        std: error in met
    """
    abX, stdX = get_elem_data(X, df)
    abH_sun = solar_abund(1)
    met = abX - abH_sun
    
    return met, stdX

# In[Functions for getting data]:

def get_elem_data(X, df):
    """
    Returns the median abundance and standard deviation
    for a given element number from the out2table data,
    df, for some star.
    Gets all ionization states of the element.
    """
    elemdf = df.drop(df[(df.elem.astype('int') != int(X))].index)
    abX = np.median(elemdf.abund.values)
    stdX = np.std(elemdf.abund.values)
    
    return abX, stdX

def get_elems(star):
    """
    Given a star's name, 
    returns an array of all included element numbers
    in its out2 file.
    """
    file = out2abs_dir + star + '.out2'
    df = out2_abfind(file)
    return np.array(list(set(np.round(df.elem.values, 1))))

def solar_abund(X):
    """
    Given the element number X, return the solar
    abundance of the element.
    Ignores ionization state.
    """
    z = int(X)
    #elem = el_sym[z-1]
    ab = solar_ab[z-1]
    return ab #, elem

def get_eps(star, X):
    """
    Returns the EP values of X for star
    """
    file = out2abs_dir + star + '.out2'
    df = out2_abfind(file)
    elemdf = df.drop(df[(df.elem.astype('int') != int(X))].index)
    
    return elemdf.ep.values

# In[]:
    
def elem_all_data(stars, X):
    for star in stars:
        file = out2abs_dir + star + '.out2'
        df = out2_abfind(file)
        abX, stdX = get_elem_data(X, df)
        el = el_sym[int(X)-1]
        print(f"""{star}:
    {el} Abundance is {abX}
    {el} STD is {stdX}
    **********************************""")

# In[Make the parameter files for the first normal run of MOOG]:

def make_other_pars(stars):
    for star in stars:
        with open(star_me_file) as f:
            alltext = f.read()
            newtext = alltext.replace('star', star)
            f.close()

        par_file = open(otherpars_dir + star + '.par', 'w')
        par_file.write(newtext)
        par_file.close()
        
    return newtext

# In[Function to add a given element's ews to the hyperfine linelist]:

def make_hfdf_ew(star, elnum):
    """
    Given a star and element number, reads the stars .ew file
    and writes the equivalent widths for hyperfine elements
    into a new .ew text file for the star.
    
    Also returns the updated dataframe for hyperfine lines.
    """
    ewfile = ews_dir + star + '.ew'
    ewdf = pd.read_csv(ewfile, skiprows = 1, usecols = [0,1,2,3,4],
                       names = ['wave', 'elem', 'col3', 'col4', 'ew', 'comments'], 
                       delim_whitespace = True)
    
    # Cull both dataframes to only include the given element number
    ewdf_el = ewdf.drop(ewdf[(ewdf.elem.astype('int') != int(elnum))].index).to_numpy()
    
    hfdf_el = hfdf.drop(hfdf[(hfdf.iso.astype('int') != int(elnum))].index).reindex()
    
    # Get the positive wavelengths from hfdf_el, the start of each blend
    startwaves = hfdf_el.loc[(hfdf_el['wave'] > 0)].copy()
    
    # find row indices in startwaves where the wavelength is close to
    # the original linelist wavelengths and update ew
    tol = 0.5
    for i in startwaves.wave.index:
        for row in ewdf_el:
            if (startwaves.wave[i] < row[0] + tol) and (startwaves.wave[i] > row[0] - tol):
                hfdf_el.ewcol[i] = row[4]
                startwaves.ewcol[i] = row[4]
    
    # Trim out all the wavelength blends that did not have an ew
    indices = startwaves.index.to_list()
    for i in indices:
        if np.isnan(hfdf_el.ewcol[i]):
            try: hfdf_el = hfdf_el.drop(labels = np.arange(i, indices[indices.index(i)+1], 1), axis = 0)
            except: hfdf_el = hfdf_el.truncate(after = i-1)
         
    # Write out to a new .ew text file
    element = el_sym[int(elnum)-1]
    
    hfdf_el.to_string(hfews_dir + 'hf_' + element.lower() + '_' + star + '.ew', 
                      col_space = [9,9,9,9,9,7,7], index = False, na_rep = '', header = None)
    
    with open(hfews_dir + 'hf_' + element.lower() + '_' + star + '.ew') as f:
        alltext = f.read()
        f.close()
    
    newtext = star + '\n' + alltext

    file = open(hfews_dir + 'hf_' + element.lower() + '_' + star + '.ew', 'w')
    file.write(newtext)
    file.close()
    
    return hfdf_el

# In[]:
    
def hfel_update_abunds(star, elnum):
    """
    Updates the original abundances file with the hyperfine blend
    abundance of a given element.
    Assumes the hyperfine out2 file has already been created.
    """
    orig_out2 = out2abs_dir + star + '.out2'
    
    element = el_sym[int(elnum)-1]
    hf_out2 = out2hfabs_dir + star + '_' + element.lower() + '.out2'
    
    hf_tab = out2_abfind(hf_out2)
    hf_tab.reset_index(drop=True, inplace=True)
    orig_tab = out2_abfind(orig_out2)

    # Find which row indices in the original out2 file table are for
    # the given element
    el_inds = orig_tab.loc[(orig_tab.elem.astype('int') == int(elnum))].copy().index.to_list()
    
    # update the original table with the hyperfine values
    j = 0
    for i in el_inds:
        orig_tab.wave[i] = hf_tab.wave[j]
        orig_tab.elem[i] = hf_tab.elem[j]
        orig_tab.ep[i] = hf_tab.ep[j]
        orig_tab.logGF[i] = hf_tab.logGF[j]
        orig_tab.EWin[i] = hf_tab.EWin[j]
        orig_tab.logRWin[i] = hf_tab.logRWin[j]
        orig_tab.abund[i] = hf_tab.abund[j]
        orig_tab.delavg[i] = hf_tab.delavg[j]
        j += 1
    
    # overwrite the old out2 file with the new data
    # this does change the format of the out2 file into just
    # the data values, without the words it had before
    orig_tab.to_string(out2abs_dir + star + '.out2',
                       justify = 'inherit', index = False, na_rep = '')
    
    return orig_tab
            
# In[Replace old abund results with new hyperfine ones]:

def update_all_abunds(stars):
    """
    Assumes all the hyperfine blend out2 files have been made.
    Updates all of those abundances in the original file with
    the new hyperfine abundances from MOOG blend.
    """            
    for star in stars:
        for el in get_elems(star):
            if int(el) in hf_elems:
                try: hfel_update_abunds(star, el)
                except: pass

# In[Function to obtain solar isotopic fractions from tab3.txt]:

def get_isofracs(X):
    """
    For some element number X (including ionization), 
    returns the solar isotopic recripocal percentages for X (second column),
    along with the corresponding isotope numbers (first column).
    
    X must be in a format like 26.0 or 63.1
    
    Update this to use the r_iso_tab for X >= 30
    """
    iso_fracs = np.array([' ',' '])
    
    if X < 30:
        xrows = iso_tab.drop(iso_tab[iso_tab.element != el_sym[int(X)-1]].index)
        
        for i in xrows.index:
            if len(str(xrows.A[i])) == 2:
                isonum = str(X)+'0'+str(xrows.A[i])
            elif len(str(xrows.A[i])) == 3:
                isonum = str(X) + str(xrows.A[i])
            invfrac = str(np.round(100/xrows.solarfrac[i], 2))
            if np.isfinite(float(invfrac)):
                iso_fracs = np.vstack((iso_fracs, np.array([isonum,invfrac])))
    
    else:
        xrows = r_iso_tab.drop(r_iso_tab[r_iso_tab.element != el_sym[int(X)-1]].index)
        
        for i in xrows.index:
            if len(str(xrows.A[i])) == 2:
                isonum = str(X)+'0'+str(xrows.A[i])
            elif len(str(xrows.A[i])) == 3:
                isonum = str(X) + str(xrows.A[i])
            nr_sum = np.sum(xrows['N(r)'].to_numpy())
            solarfrac = (xrows['N(r)'][i]) / nr_sum
            invfrac = str(np.round(1/solarfrac, 2))
            if np.isfinite(float(invfrac)):
                iso_fracs = np.vstack((iso_fracs, np.array([isonum,invfrac])))
        
    return iso_fracs[1::]

# In[]:
    
def make_par(star, el):
    """
    Write out a parameter file for a given star and
    hyperfine element. The .par file is to be run
    using MOOG with the blends driver.
    """
    with open(general_hf_file) as f:
        alltext = f.read()
        newtext = ((alltext.replace('star', star)).replace('elemnum', str(int(el)))).replace('elem', el_sym[int(el)-1].lower())
        f.close()
        
    isofracs = get_isofracs(el)
    isotxt = '\nisotopes' + 6*' ' + str(len(isofracs)) + 9*' ' + '1\n'
    
    for row in isofracs:
        isotxt += '  ' + str(row[0]) + 6*' ' + str(row[1]) + '\n'
    
    newtext += isotxt
    par_file = open(pars_dir + 'hf_' + el_sym[int(el)-1].lower() +
                    '_' + star + '.par', 'w')
    par_file.write(newtext)
    par_file.close()
    
    return newtext

# In[]:
    
# Make all parameter files for a list of stars
def make_pars(stars):
    for star in stars:
        for el in get_elems(star):
            if int(el) in hf_elems:
                make_par(star, el)
    
# For stars that have been measured and their .ew files have been
# copied over, this creates all the .ew files for its hyperfine
# elements.
def make_all_hfews(stars):
    for star in stars:
        for el in get_elems(star):
            if int(el) in hf_elems:
                make_hfdf_ew(star, el)
                  
# In[AFTER MEASUREMENTS HAVE BEEN MADE]:
   
# all except K1084 and K462 that we already did
allstars = ['1-38', 'C30131_0829', 'C30247_0911', 'K341', 'K386', 'K479', 'K583', 
         'M15-C29523_1114', 'M15-G30482_1054', 'M15-K853','M56_I_22','II-64',
         'K731_I-63','K934_I-62','s3', 'M15-pr109', 'M15C29294_1631', 
         'M15C29589_1223', 'M15C30027_0019', 'M15C30599_2226', 'M15C31264_1605', 
         'M15_I-62', 'M15_I-63', 'M15_III-59', 'M15_IV-62', 'M15_K431', 'M15_K490', 
         'M15_K969', 'm15c29413x1023']

measured = ['K462', 'K1084', '1-38', 'C30131_0829', 'C30247_0911', 'K341', 'K386', 'K479', 
            'K583', 'M15-C29523_1114', 'M15-G30482_1054', 'M15-K853', 'M15C29589_1223']
         
notmeas = ['M15-pr109', 'M15C29294_1631', 'M15C30027_0019', 
           'M15C30599_2226', 'M15C31264_1605', 'M15_I-62', 'M15_I-63', 
           'M15_III-59', 'M15_IV-62', 'M15_K431', 'M15_K490', 'M15_K969', 'm15c29413x1023']

#make_all_hfews(measured)
#make_pars(measured)
# copy all into stravinsky, run moog, then copy out2 files into here
# In[]:
    
#update_all_abunds(measured)
    
def plot_ep_ab(star):
    
    plt.figure(linewidth = 20)
    plt.title(star)
    plt.xlabel('Excitation Potential')
    plt.ylabel('Abundance')
    
    df = out2_abfind(out2abs_dir + star + '.out2')
    nonzero_ep = df.drop(df[(df.ep.astype('float') == 0.) | (df.elem.astype('float') != 26.)].index)
    eps = nonzero_ep.ep.values
    abunds = nonzero_ep.abund.values
    
    plt.scatter(eps, abunds)
    plt.show()
    
    return

def plot_star_XYs(stars, X1, Y1, X2, Y2):
    """
    Given a set of stars, gets the metallicity XY
    for each star and plots them in a scatter plot
    with error bars.
    """
    plt.figure(linewidth = 20)
    #plt.title('M15 Metallicity')
    plt.axis('equal')
    plt.xlabel('['+el_sym[int(X1)-1]+'/'+el_sym[int(Y1)-1]+']')
    plt.ylabel('['+el_sym[int(X2)-1]+'/'+el_sym[int(Y2)-1]+']')
    for star in stars:
        table = out2_abfind(out2abs_dir + star + '.out2')
        met1, std1 = met_XY(table, X1, Y1)
        met2, std2 = met_XY(table, X2, Y2)
        plt.errorbar(met1, met2, yerr = std2, xerr = std1, label = star,
                     fmt = 'o', markersize = 8, capsize = 10, capthick = 1.5)
    plt.legend()
    plt.show()
    
    return

# In[These are the main functions you'll want to use]:
    
def main(stars):
    input('.ew files have been copied this computer?')
    print('Okay, making .par files for these stars...')
    make_other_pars(stars)
    input('Copy those .par files from here into M15/abund. Done?')
    input('Now run MOOG on those, and then copy the .out2 files to this computer. Done?')
    
    print('Alright, making the hyperfine .ew files and the hyperfine .par files.')
    make_all_hfews(stars)
    make_pars(stars)
    input('Now copy those into M15/hfew. Done?')
    input('Okay, run MOOG for those stars in M15/hfew. Done?')
    
    input('Now copy those .out2 files into out2_hfabunds. Done?')
    update_all_abunds(stars)
    
    print('All done. See if the plots work!')
    
    return
    
def plot_mgeu(stars):
    
    plt.figure(linewidth = 20)
    plt.title('M15 Mg vs Eu Abundances')
    plt.axis('equal')
    plt.xlabel('A(Mg)')
    plt.ylabel('A(Eu)')
    for star in stars:
        table = out2_abfind(out2abs_dir + star + '.out2')
        ab_eu, std_eu = get_elem_data(63, table)
        ab_mg, std_mg = get_elem_data(12, table)
        
        if star == 'M15C30599_2226' or star == 'M15C30027_0019':
            plt.errorbar(ab_mg, ab_eu, yerr = std_eu, xerr = std_mg,
                         fmt = 'ro', markersize = 6, capsize = 5, capthick = 1.5)
        else:
            plt.errorbar(ab_mg, ab_eu, yerr = std_eu, xerr = std_mg,
                         fmt = 'ko', markersize = 6, capsize = 5, capthick = 1.5)
    plt.show()
    
    return
