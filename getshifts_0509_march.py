import numpy as np
from numpy import *
from astropy import *
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import imresize
from scipy.optimize import curve_fit as cfit
from scipy.optimize import minimize as cmin

################################################################################################
#######################################  NOTE!!!   #############################################
###  Make sure that you have a subdirectory named "figs" in your working directory where you are
###  running this code along with all of the images and region file that are needed for its
###  execution
################################################################################################
################################################################################################

#***********************************************************************************************

################################################################################################
#######################################    Begin   #############################################
#########################   Input values/file names from user!!!   #############################
################################################################################################
################################################################################################

plotter = True      # Option to blot brightness profiles
pixfrac = 0.8       # Pixfrac used for thes drizzled images
scale   = 0.05      # size of pixels as measured by "arcsec/pixel"

iter_size = 100   # Number of iterations for the monte carlo simulation that perturbs brightness
                  # profiles with there measured uncertainties in order to measure the median shift 
                  # and the uncertainty of that shift measurement which is determined from the standard 
                  # deviation from the simulation results

xcen = 505.73   # X - Pixel location for the center of the remnant
ycen = 504.17   # Y - Pixel location for the center of the remnant


################     Input image files for the routiner       #####################################
# Image for epoch 1 science image that is star subracted
ep1_img_dat = 'ep1_ha_p05_arc_0p8_pfrac_drc_sci_star_sub.fits'
# Inverse variance image from astrodrizzle output for epoch 1 ("ep1*drc_wht.fits")
ep1_img_wht = 'ep1_ha_p05_arc_0p8_pfrac_drc_wht.fits'    
# Image for epoch 3 science image that is star subracted
ep2_img_dat = 'ep3_ha_p05_arc_0p8_pfrac_scale0p05_drc_sci_star_sub.fits'    
# Inverse variance image from astrodrizzle output for epoch 3 ("ep3*drc_wht.fits")
ep2_img_wht = 'ep3_ha_p05_arc_0p8_pfrac_scale0p05_drc_wht.fits'   
# Name of the region file containing the extraction apertures for the brighness profiles
reg_file_name = 'paper_outer_phys.reg'



################################################################################################
#######################################    End     #############################################
#########################   Input values/file names from user!!!   #############################
################################################################################################
################################################################################################

def ellipse_finder(x0,pos_array):
    
    ret_val = np.sqrt(sum((x0[1]**2.*(((pos_array[0,:]-x0[2])*np.cos(x0[4])) + \
                                       (pos_array[1,:]-x0[3])*np.sin(x0[4]))**2. + \
                           
                           x0[0]**2.*(((pos_array[0,:]-x0[2])*np.sin(x0[4])) - \
                           (pos_array[1,:]-x0[3])*np.cos(x0[4]))**2.) - \
                          
                           x0[0]**2.*x0[1]**2.)**2.)
    return ret_val

#***********************************************************************************************

def fit_profiles(x0, ep1_shifted, ep2_dat, ep1_weight_dat, ep2_weight_dat):
#     val_ret = float(sum(((x0[1] * np.sqrt(ep1_shifted) + x0[0]) - np.sqrt(ep2_dat))**2. / 
#                         (np.sqrt(np.sqrt(ep1_weight_dat)) + np.sqrt(np.sqrt(ep2_weight_dat)))))
    val_ret = float(sum(((x0[1] * (ep1_shifted) + x0[0]) - (ep2_dat))**2. / 
                        (((ep1_weight_dat)) + ((ep2_weight_dat)))))
#     print val_ret
    return val_ret
    

################################################################################################
#####################################   Begin          #########################################
##########    Sub-routine to find the shift for a given set of 1-d brightness profiles    ######
################################################################################################
################################################################################################

def shifter(ax_ep1, ax_ep2, ax_ep1_sc, ep1_sc, ep1_weight_sc, ep2_holder, ep2_weight_dat, m):
    shift = np.arange(0,9,1)*1.+2.
    shift_size = len(shift)
    stat = np.zeros(shift_size)
    stat_num = np.zeros(len(ax_ep2))
    stat_denom = np.zeros(len(ax_ep2))
    ax_ep1_hold = ax_ep1_sc
    for t in range (0, shift_size):
        ax_ep1_sc = np.arange(min(ax_ep1), max(ax_ep1), .01)
        ax_ep1_sc = ax_ep1_sc + shift[t]

        for p in range(0, size(ep2_holder)):
            indx_dat3 = np.where(around(ax_ep1_sc, decimals = 2) == ((p + 1) * 1.0))
            if size(indx_dat3) == 0:
                stat_num[p]=np.sqrt(size(ax_ep2) * 1. - 1.)
                stat_denom[p] = 1.
            else:
#                 stat_num[p] = (rat2[t] * ep1_sc[(indx_dat3)] - ep2_holder[p])
#                 stat_denom[p] = (rat[t]**2. * ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
                stat_num[p] = (ep1_sc[(indx_dat3)] - ep2_holder[p])
                stat_denom[p] = (ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
        stat[t] += sum(stat_num**2. / stat_denom)
#     print(stat)
    res1 = shift[where(stat == min(stat))]
#     print('res1='+str(res1))
    del shift
    shift = np.arange(res1-1.,res1+2.,.1)
    shift_size = len(shift)
    stat = np.zeros(shift_size)
    stat_num = np.zeros(len(ax_ep2))
    stat_denom = np.zeros(len(ax_ep2))

    for t in range (0, shift_size):
        ax_ep1_sc = np.arange(min(ax_ep1), max(ax_ep1), .01)
        ax_ep1_sc = ax_ep1_sc + shift[t]

        for p in range(0, len(ep2_holder)):
            indx_dat3 = np.where(around(ax_ep1_sc, decimals = 2) == ((p + 1) * 1.0))
            if size(indx_dat3) == 0:
                stat_num[p]=np.sqrt(size(ax_ep2) * 1. - 1.)
                stat_denom[p] = 1.
            else:
#                 stat_num[p] = (rat2[t] * ep1_sc[(indx_dat3)] - ep2_holder[p])
#                 stat_denom[p] = (rat[t]**2. * ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
                stat_num[p] = (ep1_sc[(indx_dat3)] - ep2_holder[p])
                stat_denom[p] = (ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
        stat[t] += sum(stat_num**2. / stat_denom)
    res2 = shift[where(stat == min(stat))]
#     print('res2='+str(res2))
    del shift
    shift = np.arange(res2 - .1, res2 + .2, .01)
    shift_size = len(shift)
    stat = np.zeros(shift_size)
    stat_num = np.zeros(len(ax_ep2))
    stat_denom = np.zeros(len(ax_ep2))

    for t in range (0, shift_size):
        ax_ep1_sc = np.arange(min(ax_ep1), max(ax_ep1), .01)
        ax_ep1_sc = ax_ep1_sc + shift[t]

        for p in range(0, size(ep2_holder)):
            indx_dat3 = np.where(around(ax_ep1_sc, decimals = 2) == ((p + 1) * 1.0))
            if size(indx_dat3) == 0:
                stat_num[p]=np.sqrt(size(ax_ep2) * 1. - 1.)
                stat_denom[p] = 1.
            else:
#                 stat_num[p] = (rat2[t] * ep1_sc[(indx_dat3)] - ep2_holder[p])
#                 stat_denom[p] = (rat[t]**2. * ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
                stat_num[p] = (ep1_sc[(indx_dat3)] - ep2_holder[p])
                stat_denom[p] = (ep1_weight_sc[(indx_dat3)] + ep2_weight_dat[p])
        stat[t] += sum(stat_num**2. / stat_denom)
    res3 = shift[where(stat == min(stat))]
    
    return res3 * speed_conversion, stat[where(stat == min(stat))]

################################################################################################
#####################################   End of         #########################################
##########    Sub-routine to find the shift for a given set of 1-d brightness profiles    ######
################################################################################################
################################################################################################

#***********************************************************************************************

################################################################################################
################################################################################################
################################    Main Program     ###########################################
################################################################################################
################################################################################################

#############   Open output file for the latex deluxe table     ###############################
tex_out=open(reg_file_name[:-4] + '_p0_' + str(scale)[3:] + '_arcsec_tex_format_' + str(iter_size) + '_iter_part1.txt','w+')
tex_out.write(r'\begin{deluxetable*}{ccccccc}[H]' + ' \n')
tex_out.write(r'\tablecolumns{7}' + ' \n')
tex_out.write(r'\tablewidth{0pc}' + ' \n')
tex_out.write(r'\tabletypesize{\scriptsize}' + ' \n')
tex_out.write(r'\tablecaption{Proper Motion Measurements for SNR \remnant{}}' + ' \n')
tex_out.write(r'\tablehead{\colhead{ID} & \colhead{PA[Deg]} & \colhead{Radius[$^{\prime\prime}$]} &  \colhead{Shift [mas]} & \colhead{$\chi^2$ } & \colhead{d.o.f.} & \colhead{${\rm{V}_r}$ [km/s]$^{^{\left(1 \right)}}$}}' + ' \n')
tex_out.write(r'\startdata' + ' \n')
               
#############   Open output file for the latex raw results    ###################################
file_out_raw=open(reg_file_name[:-4] + '_p0_' + str(scale)[3:] + '_arcsec_raw_format' + str(iter_size) + '_iter_part1.txt','w+')
file_out_raw.write('region number  pos_angle[deg]  radius[arcsec]   shift[mas]     velocity[km/s]   1-sig[km/s]   ch-sq    dof tot_counts[cps] \n') 


##########################      Constants      #################################################
lmc_dist = 50000. #parsec
day_to_sec = 24. * 3600.
time_base =  3669. # time between epochs measured in days
pc_to_km = 3.086e13
speed_conversion = lmc_dist * scale * pc_to_km / (206265. * time_base * day_to_sec)  # converts shift in pixels 
                                                                                    # to speed in km/sec

#########   Determine the Multiplicitive constant to apply to weight images due to the correlation 
#########   of noise that is added in the drizzling process that artificially increases the uncertainty 
#########   of a given pixel's flux.  See : A. S. Fruchter1 and R. N. Hook (2002) page 151
#########   http://adsabs.harvard.edu/abs/2002PASP..114..144F

# Bibtex entry
#
# @ARTICLE{2002PASP..114..144F,
#    author = {{Fruchter}, A.~S. and {Hook}, R.~N.},
#     title = "{Drizzle: A Method for the Linear Reconstruction of Undersampled Images}",
#   journal = {\pasp},
#    eprint = {astro-ph/9808087},
#  keywords = {Methods: Data Analysis, Techniques: Photometric},
#      year = 2002,
#     month = feb,
#    volume = 114,
#     pages = {144-152},
#       doi = {10.1086/338393},
#    adsurl = {http://adsabs.harvard.edu/abs/2002PASP..114..144F},
#   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
# }

s_fh = scale / 0.05   # fraction scale size is to natiove 0.05" pixels of ACS
r_fh = pixfrac / s_fh
if (r_fh < 1.):
    fh_constant = 1. / (1. - (r_fh / 3.))
if (r_fh >= 1.):
    fh_constant = r_fh / (1. - (1. / (r_fh * 3.)))

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



################     Read in the fits images from section on top ###################################            
ep1_img = fits.getdata(ep1_img_dat,0)
ep2_img = fits.getdata(ep2_img_dat,0)
ep1_weight_img = fits.getdata(ep1_img_wht,0)
ep2_weight_img = fits.getdata(ep1_img_wht,0)
hdulist = fits.open(ep1_img_dat)
ep1_mask = ep1_img*0
ep2_mask = ep1_img*0

################    Read in the regions for the extraction process   ##############################            
reg_file = open(reg_file_name,'r')
reg_lines = reg_file.readlines()
xpos=np.zeros(len(reg_lines)-1)
ypos = xpos * 0.
szx = xpos * 0.
szy = xpos * 0.
szx2 = xpos * 0.
xval = xpos * 0.
yval = xpos * 0.
angle_reg = xpos * 0.
rad_arr = xpos * 0.
rad_err = xpos * 0.


for i in range(1,len(reg_lines)):
    line = reg_lines[i].split(',')
    if line[0][:4]=='box':
        xpos[i - 1] = float(line[0][4:]) - 1.
    else:
        xpos[i - 1] = float(line[0][7:]) - 1.
    ypos[i - 1] = float(line[1]) - 1.
    szx[i - 1] = float(line[2]) / 2.# + 4.
    szx2[i - 1] = float(line[2]) / 2. + 10.
    szy[i - 1] = float(line[3]) / 2.
    angle_reg[i - 1] = float(line[-1][:-2])
reg_file.close()

pos_angle = np.arctan2((ypos - ycen),(xpos - xcen)) * 180 / (np.pi) - 90.0
pafix = where(pos_angle <= 0.0)
pos_angle[pafix] = pos_angle[pafix]+ 360.0


# check to makesure the number of regions corresponds to the max of the mask array
nsz = len(xpos)
the = angle_reg * 1.
xarray, yarray = np.meshgrid(arange(ep1_img.shape[0]), arange(ep1_img.shape[1])) # Make 2d arrays to denote 
                                                                                 # x and y locations of pixels

for i in range(1, nsz + 1):
    print('working on region ' + str(i) + ' out of ' + str(nsz))
        
        
    # Make the rim masks for the measurement
    
    sinthe = np.sin(((270. - angle_reg[i - 1])) / 180. * np.pi)
    costhe = np.cos((270. - angle_reg[i - 1]) / 180. * np.pi)
    sinthe2 = np.sin((-(270. - angle_reg[i - 1])) / 180. * np.pi)

    
    xarray2 = xarray*0. + (((xarray - xpos[i - 1])) * costhe - \
          ((yarray - ypos[i - 1])) * sinthe)

    yarray2 = xarray*0. + (((yarray - ypos[i - 1])) * costhe + \
              ((xarray - xpos[i - 1])) * sinthe)

    xarray3 = xarray*0. + (((xarray - xpos[i - 1])) * costhe - \
              ((yarray - ypos[i - 1])) * sinthe)

    yarray3 = xarray*0. + (((yarray - ypos[i - 1])) * costhe + \
              ((xarray - xpos[i - 1])) * sinthe)
    yarray4 = xarray*0. + (((yarray - ypos[i - 1])) * costhe + \
              ((xarray - xpos[i - 1])) * sinthe)

    
    indx2 = np.where((xarray3 <= 1.*szy[i-1]) & (xarray3 >= (-1.*szy[i-1])) & \
                   (yarray3 <= 1.*szx2[i-1]) & (yarray3 >= (-1.*szx2[i-1])))
    
    indx = np.where((xarray2 <= 1.*szy[i-1]) & (xarray2 >= (-1.*szy[i-1])) & \
                   (yarray2 <= 1.*szx[i-1]) & (yarray2 >= (-1.*szx[i-1])))
    
    
    exc_indx = np.where(((xarray2 > 1.*szy[i-1]) | (xarray2 < (-1.*szy[i-1]))) | \
                        ((yarray2 > 1.*szx[i-1]) | (yarray2 < (-1.*szx[i-1]))))
    exc_indx2 = np.where(((xarray3 > 1.*szy[i-1]) | (xarray3 < (-1.*szy[i-1]))) | \
                         ((yarray3 > 1.*szx2[i-1]) | (yarray3 < (-1.*szx2[i-1]))))
    
    
    xarray2[exc_indx] = xarray2[exc_indx] * 0. - 500.
    yarray2[exc_indx] = yarray2[exc_indx] * 0. - 500.
    xarray3[exc_indx2] = xarray3[exc_indx2] * 0. - 500.
    yarray3[exc_indx2] = yarray3[exc_indx2] * 0. - 500.
    yarray4 = yarray3*0. + yarray3
    ep2_mask[indx] = ep2_mask[indx] * 0. + i
    ep1_mask[indx2] = ep1_mask[indx2] * 0. + i
    fits.writeto('rim_mask_ep1_p0' + str(scale)[3:] + \
                 '_arcsec.fits', ep1_mask, hdulist[0].header, clobber = 'yes')
    
    fits.writeto('rim_mask_ep2_p0' + str(scale)[3:] + \
                 '_arcsec.fits', ep2_mask, hdulist[0].header, clobber = 'yes')

    upper = int(round(amax(yarray2[(indx)])))
    lower = int(round(amin(yarray2[(indx)])))
    upper_ep1 = int(round(amax(yarray3[(indx2)])))
    lower_ep1 = int(round(amin(yarray3[(indx2)])))
    
    ax_ep1 = 1. * np.arange(lower_ep1+1,upper_ep1+1,1.)
    ep1_dat = ax_ep1 * 0
    ep1_weight_dat = ax_ep1 * 0.
   
    ax_ep2 = 1. * np.arange(lower+1,upper+1,1.)
    ep2_dat = ax_ep2 * 0.
    ep2_weight_dat = ep2_dat * 0.
    
    

    for k in range (0,len(ax_ep1)):

        newindx2 = np.where((yarray3 > (float(k) + lower_ep1)) & 
                            (yarray3 <= (float(k + 1) + lower_ep1)) & 
                            (ep1_weight_img > 0.) & 
                            (ep1_img > 0.))
        if len(newindx2) > 0:
            ep1_dat[k] = sum(ep1_img[(newindx2)] * ep1_weight_img[newindx2]) / \
                             sum(ep1_weight_img[newindx2])
            ep1_weight_dat[k] = 1. / sum(ep1_weight_img[newindx2])# / fh_constant**2.)
        else:
            ep1_dat[k] = 0.
            ep1_weight_dat[k] = std(ep1_img[where(ep1_mask == i)])

    for k in range (0,len(ax_ep2)):

        newindx = np.where((yarray3 > (float(k) + lower)) & 
                            (yarray3 <= (float(k + 1) + lower)) & 
                            (ep1_weight_img > 0.) & 
                            (ep1_img > 0.)) 
        if len(newindx) > 0:
            ep2_dat[k] = sum(ep2_img[(newindx)] * ep2_weight_img[newindx]) / \
                             sum(ep2_weight_img[newindx])
            ep2_weight_dat[k] = 1. / sum(ep2_weight_img[newindx])# / fh_constant**2.)
        else:
            ep2_dat[k] = 0.
            ep2_weight_dat[k] = std(ep2_img[where(ep2_mask == i)])

    
    ax_ep2 += upper
    ax_ep1 += upper
    ############################################################################################
    #################################   Begin          #########################################
    ############    Monte Carlo Routine that will find shift and its uncertainty  ##############
    ############################################################################################
    ############################################################################################

    stat_monte = np.zeros(iter_size)
    rad_monte = np.zeros(iter_size)
    r_inner_monte = np.zeros(iter_size)
    r_half_monte = np.zeros(iter_size)
    rat_monte = np.zeros(iter_size)
    rat2_monte = np.zeros(iter_size)
    shift_monte = np.zeros(iter_size)
    random_num = np.random.randn((iter_size - 1) * size(ep1_dat))
    random_num2 = np.random.randn((iter_size - 1) * size(ep2_dat))

    for m in range(0, iter_size):
        
        if ((m % 100) == 0):
            print ('Region ' + str(i) + ' iteration ' + str(m + 1))
        if m < iter_size - 1:
            ep1_holder = ep1_dat + np.sqrt(ep1_weight_dat) * random_num[size(ep1_dat) * \
                         m:size(ep1_dat) * m + size(ep1_dat)]

            ep2_holder = ep2_dat + np.sqrt(ep2_weight_dat) * random_num2[size(ep2_dat) * \
                         m:size(ep2_dat) * m + size(ep2_dat)]
        else:
            ep1_holder = ep1_dat
            ep2_holder = ep2_dat

        ax_ep1_sc = np.arange(100. * (amax(ax_ep1[np.where(ep1_dat == ep1_dat)]) - \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)])) + 1.) / 100. + \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)])
        if m > 0:
            del f 
            del f2
        f = interp1d(ax_ep1[np.where(ep1_holder == ep1_holder)], ep1_holder[np.where(ep1_holder == ep1_holder)])#, kind = 'cubic')
        
        ep1_sc = f(ax_ep1_sc)
        f2 = interp1d(ax_ep1, ep1_weight_dat, kind = 'cubic')
        ep1_weight_sc = f2(ax_ep1_sc)

        
        ########################################################################################
        #####################   Call out to shifter routine! ###################################        

        shift_monte[m], stat_monte[m] = shifter(ax_ep1, ax_ep2, ax_ep1_sc, ep1_sc, ep1_weight_sc, \
                                                              ep2_holder, ep2_weight_dat, m)
        ax_ep2_sc = np.arange((len(ax_ep2)-1)*100+1)/100.+ax_ep2[0]
        f3 = interp1d(ax_ep2,ep2_holder, kind = 'cubic')

        ep2_sc = f3(ax_ep2_sc)
        r_inner_monte[m] = ax_ep2_sc[where((ep2_sc == max(ep2_sc[where(ax_ep2_sc > 2.)])))][0]
        r_half_hold = (ax_ep2_sc[where((ax_ep2_sc > r_inner_monte[m]) & 
                 (ep2_sc <= (0.5*(max(ep2_sc) - .005))))])
        if size(r_half_hold)==0:
            ep2_new_sc = ep2_sc*0.
            ep2_new_sc += ep2_sc**2.
            r_half_hold = (ax_ep2_sc[where((ax_ep2_sc > r_inner_monte[m]) & 
                 (ep2_new_sc <= (0.5*(max(ep2_new_sc) - min(ep2_new_sc)))))])
        if size(r_half_hold)==0:
            r_half_hold = ax_ep2[where(ax_ep2 >= (4 + r_inner_monte[m]))]
        r_half_monte[m] = r_half_hold[0]
        rad_monte[m] = sqrt((4.*r_half_monte[m]**2.-r_inner_monte[m]**2.)/3.)

        ########################################################################################    
        ########################################################################################


    #############################################################################################
    ##################################   End of         #########################################
    #############    Monte Carlo Routine that will find shift and its uncertainty  ##############
    #############################################################################################
    #############################################################################################



######################    Determine the best fit values from the simulation  ###################            
    shift_sort = shift_monte.argsort()   
    stat_monte = stat_monte[shift_sort]
    
    
    if (float(iter_size) % 2. == 0):
        stat_report = stat_monte[iter_size / 2 - 1]
    else:
        stat_report = stat_monte[(iter_size - 1) / 2 - 1]

    print('v = ' + str(int(median(shift_monte))) + ' +/- ' + str(int(std(shift_monte))) + ' km/sec')
    ######################    Plot 1-D spatial brightness profiles for each region    ##############    
    
    plt.clf()
    ax_ep1_sc_old = np.arange(100. * (amax(ax_ep1[np.where(ep1_dat == ep1_dat)]) - \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)])) + 1.) / 100. + \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)])
    ax_ep1_sc = np.arange(100. * (amax(ax_ep1[np.where(ep1_dat == ep1_dat)]) - \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)])) + 1.) / 100. + \
                    amin(ax_ep1[np.where(ep1_dat == ep1_dat)]) + \
                    round(100. * shift_monte[-1] / speed_conversion)/100.
    ep1_shifted = ep2_dat * 0.
    ep1_shifted_weight = ep2_dat * 0.
    for p in range(0, size(ep2_dat)):
        indx_dat = np.where(around(ax_ep1_sc, decimals = 2) == ((p+1) * 1.0))
        ep1_shifted[p] = ep1_sc[indx_dat]
        ep1_shifted_weight[p] = ep1_weight_sc[indx_dat]
    
    
    x0 = [0.001, 1.0]
    bnds = ((-.001, .5),\
            (.1, 3.0))
    fit_result = cmin(fit_profiles, \
               x0, \
               args = (ep1_shifted, ep2_dat, ep1_shifted_weight, ep2_weight_dat), \
               method = 'SLSQP')#, \
#                bounds = bnds)   

    ep1_shifted = fit_result['x'][0] + ep1_shifted * fit_result['x'][1]
    ep1_sc_new = fit_result['x'][0] + ep1_sc * fit_result['x'][1]
    ep1_dat_new = fit_result['x'][0] + ep1_dat * fit_result['x'][1]
    stat_print = fit_result['fun'] 

    
    
    
    ax_ep2_sc = np.arange((len(ax_ep2)-1)*100+1)/100.+ax_ep2[0]
    f3 = interp1d(ax_ep2,ep2_dat, kind = 'cubic')
        
#     ep2_sc = f3(ax_ep2_sc)
#     r_inner = ax_ep2_sc[where((ep2_sc == max(ep2_sc[where(ax_ep2_sc > 10.)])))][0]
#     r_half = ax_ep2_sc[where((ax_ep2_sc > r_inner) & 
#              (ep2_sc <= (0.5*(max(ep2_sc) - .005))))]
#     if len(r_half)==0:
#         ep2_new_sc = ep2_sc*0.
#         ep2_new_sc += ep2_sc**2.
#         r_half = ax_ep2_sc[where((ax_ep2_sc > r_inner) & 
#              (ep2_new_sc <= (0.5*(max(ep2_new_sc) - min(ep2_new_sc)))))]
    r_sh = median(rad_monte)#sqrt((4.*r_half[0]**2.-r_inner**2.)/3.)
    xval[i - 1] = xval[i - 1] - (sinthe2 * (r_sh - szx[i - 1])) + xpos[i - 1] + 1.
    yval[i - 1] = (r_sh - szx[i - 1]) * costhe + ypos[i - 1] + 1
    rad_arr[i - 1] = scale * np.sqrt((xval[i - 1] - xcen)**2. + (yval[i - 1] - ycen)**2.)
    rad_err[i - 1] = scale * np.sqrt((median(r_half_monte - r_inner_monte))**2. + std(rad_monte)**2.)
    rad_err[i - 1] = int(rad_err[i - 1] * 100.) / 100.
    rad_arr[i - 1] = int(rad_arr[i - 1] * 10.) / 10.
    if rad_err[i - 1] < 0.01:
        rad_err[i - 1] = 0.01
    print('r_shock = '+str(rad_arr[i - 1])+'+/-'+str(rad_err[i - 1]) + ' arcsec')
    
#     r_sh=int(r_sh*10.)/10.
#     r_sh_low = r_sh-.5
#     r_sh_high = r_sh+.5
#     in2 = np.where(((yarray4 + upper) >= r_sh_low) & ((yarray4 + upper) <= r_sh_high))
#     if size(in2)==0:
#         in2 = np.where(yarray2==amax(yarray2))
#         yval[i-1] = yarray[where(yarray4 == yarray4[in2][0])] + 1.
#     else:
#         yval[i - 1] = yarray[where(yarray4 == yarray4[in2][len(in2) / 2])] + 1.
#     xval[i - 1] = xarray[where((xarray2 == amin(xarray2[where(xarray2 >= 0.)])))] +1.

    
    print('reduced chi^2 for region ' + str(i) + ' = ' + str(fit_result['fun'] / (len(ax_ep2) - 1.)))
    print('------------------------------------------------------------------------------')
    rat = median(rat_monte)
    rat2 = median(rat2_monte)
    if plotter==True:
#         ep1_dat=np.sqrt(ep1_dat)
#         ep2_dat=np.sqrt(ep2_dat)
#         ep1_weight_dat=np.sqrt(np.sqrt(ep1_weight_dat))
#         ep2_weight_dat=np.sqrt(np.sqrt(ep2_weight_dat))
#         ep1_sc=np.sqrt(ep1_sc)
#         ep1_weight_sc=ep1_sc=np.sqrt(np.sqrt(ep1_weight_sc))
#         ep1_sc_new=np.sqrt(ep1_sc_new)
#         ep1_shifted=np.sqrt(ep1_shifted)
#         ep1_shifted_weight=np.sqrt(np.sqrt(ep1_shifted_weight))

        plt.clf()
        plt.plot([r_sh,r_sh],[-100,10000],color='green')
        plt.plot(ax_ep1,1000. * ep1_dat_new, color = 'black')
        plt.plot(ax_ep2,1000. * ep2_dat, color = 'red')
        plt.errorbar(ax_ep2, 1000. * ep2_dat, 1000. * np.sqrt(ep2_weight_dat), color = 'red')
#         plt.plot(ax_ep1_sc, 1000. * ep1_sc_new, color = 'blue')
        plt.errorbar(ax_ep2, 1000. * ep1_shifted,1000. * np.sqrt(ep1_shifted_weight),color='blue')
        plt.errorbar(ax_ep2, 1000 * (ep2_dat-ep1_shifted), 1000. * np.sqrt(ep1_shifted_weight + ep2_weight_dat), \
                     color = 'purple', ls ='none')
        plt.plot(ax_ep1, ax_ep1 * 0., color = 'purple')

        plt.axis([min(ax_ep2) - .5,  
                  max(ax_ep2) + .5, 
                  -5.,  
                  1100 * max(np.array([max(ep2_dat),max(ep1_shifted)])) + 
                              max(np.sqrt(ep1_shifted_weight + ep2_weight_dat))]) 
        plt.xlabel('Radius[Pixels]')
        plt.ylabel('Flux[cpks]')
        plt.title('Region ' + str(i))
        plt.savefig('./figs/region_'+str(i) + '_p0' + str(scale)[3:] + '_arcsec.pdf',format='pdf')
#     print((stat_monte[-1])/(len(ax_ep2)-1)) # reduced chi squared for this regions median fit
######################    Print results to be inserted into a latex table  #####################    
    tex_out.write(str(i) + ' & ' + \
                   str(round(10.*pos_angle[i-1])/10.) + ' & ' + \
                   str(rad_arr[i - 1]) + '$\pm$' + \
                   str(rad_err[i - 1]) + ' & ' + \
                   str(round(1000.*100*scale*median(shift_monte)/(speed_conversion))/100.) + '$\pm$' + \
                   str(round(round(1000.*100*scale*std(shift_monte)/(speed_conversion))/100.)) + ' & ' + \
                   str(round(10. * float(stat_print)) / 10.) + ' & ' + \
                   str(size(ax_ep2) - 1) + ' & ' + \
                   str(round(float(median(shift_monte)))) + '$\pm$' + \
                   str(round(std(shift_monte))) + ' \\'+'\\'+ '\n')
######################    Print results in a raw formagt for cursory inspection  ###############    
    file_out_raw.write(str(i) + ' ' + \
                    str(round(10.*pos_angle[i-1])/10.) + ' ' + \
                    str(rad_arr[i - 1]) + ' ' + \
                    str(rad_err[i - 1]) + ' ' + \
                    str(round(1000.*100*scale*median(shift_monte)/(speed_conversion))/100.) + ' ' + \
                    str(round(round(1000.*100*scale*std(shift_monte)/(speed_conversion))/100.)) + ' ' + \
                    str(round(10. * float(stat_print)) / 10.) + ' ' + \
                    str(size(ax_ep2) - 1) + ' ' + \
                    str(round(float(median(shift_monte)))) + ' ' + \
                    str(round(std(shift_monte))) + '\n')    
    
######################    Plot histograms for monte carlo sim     ##############################    
    
    if ((plotter==True) and (iter_size >= 1000)):
        plt.clf()
        plt.hist(shift_monte)
        plt.savefig('./figs/hist_region_' + str(i) + '_velocites_med_fit_' + \
                     str(iter_size) + '_iters_p0' + str(scale)[3:] + '_arcsec.pdf', format = 'pdf')
        plt.clf()
        plt.hist(stat_monte)
        plt.savefig('./figs/hist_region_' + str(i) + '_chi_squared_med_fit_' + \
                     str(iter_size) + '_iters_p0' + str(scale)[3:] + '_arcsec.pdf', format = 'pdf')
    
    

    
tex_out.write(r'\enddata \n')
tex_out.write(r'\vspace{-0.3cm} \n') 
tex_out.write('tablecomments{$^1$ These velocities are the component in the plane of the sky. \\' + ' \n')
tex_out.write('Best fit values of the shift and shock speed for each extraction aperture are followed by their ' + \
        '$1\sigma$ statistical uncertainties.}' + ' \n')
tex_out.write(r'\label{tab:expansion_table}' + ' \n')
tex_out.write(r'\end{deluxetable}' + ' \n')

tex_out.close()
file_out_raw.close()
        
outfile = open(reg_file_name[:-4] + '_shock_xy.txt','w')
outfile.write('Region        x_sh        ysh' + '\n')
for i in range(0, len(xval)):
    outfile.write(str(i+1) + '\t' + str(xval[i]) + '\t' + str(yval[i]) + '\n')
outfile.close()
outfile = open(reg_file_name[:-4] + '_shock_xy.reg','w')
outfile.write(reg_lines[0])
for i in range(0, len(xval)):
    outfile.write(r'circle(' + str(xval[i]) + ',' + str(yval[i]) + ',' + r'3.0)' + '\n')
outfile.close()
