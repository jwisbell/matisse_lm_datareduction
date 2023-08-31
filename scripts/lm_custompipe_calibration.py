import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from glob import glob
from astropy.io import fits 
from scipy.special import jv, jve 


COLORS = ['firebrick','lightgreen','navy','mediumorchid','coral','cyan','k']

def read_spectrum(wl):
    f = open('../../calibrator_templates/HD120404_spectrum.txt','r')
    lines = f.readlines()
    wave, flux = [], []
    for l in lines:
        if l[0] != '#':
            data = l.split()
            wave.append(float(data[0]))
            flux.append(float(data[1]))

    myflux = np.interp(wl*1e6, wave, flux )
    return myflux



def do_calibration(targ_tpl, calib_tpl, debug=0):#targ_cflux,targ_cphase, calib_cflux, calib_cphase):
    targ_cflux = './custom_pipeline/bcd_corrflux_{tpl}.pk'.format(tpl = targ_tpl)
    targ_cphase = './custom_pipeline/bcd_cphase_{tpl}.pk'.format(tpl = targ_tpl)
    calib_cflux = './custom_pipeline/bcd_corrflux_{tpl}.pk'.format(tpl = calib_tpl)
    calib_cphase = './custom_pipeline/bcd_cphase_{tpl}.pk'.format(tpl = calib_tpl)
    
    df_targ_cflux = pd.read_pickle(targ_cflux)
    df_calib_cflux = pd.read_pickle(calib_cflux)
    
    df_targ_cphase = pd.read_pickle(targ_cphase)
    df_calib_cphase = pd.read_pickle(calib_cphase)
    
    wls = df_targ_cflux['wlarr'][0]
    true_flux = read_spectrum(wls) #needs to be in micron

    #fig = plt.figure()
    flux_vals = [[],[],[],[],[],[]]
    err_vals = [[],[],[],[],[],[]]
    t3phi_vals = [[],[],[],[]]
    t3phi_errs = [[],[],[],[]]
    mask = [True, True, True, True, True, True]
    s = np.where(np.logical_and(wls*1e6 > 4.65, wls*1e6 < 4.8) )[0]

    templates = glob('./lm_formatting/template_oifits/*.fits')
    #tpl_calib = calib_cflux.split('_')[-1].split('.pk')[0].split('T')[0] + 'T'+calib_cflux.split('_')[-1].split('.pk')[0].split('T')[1].replace('-',':')
    tpl_calib = calib_tpl.replace('-',':')

    for t in templates:
        hdu = fits.open(t)
        tpl = hdu[0].header['eso tpl start']
        
        if tpl == tpl_calib:
            #fill the file with my calibrated correlated fluxes
            #repeat them if necessary?

            #need to correct for resolution effects!!!!! 
            ######################
            ## set this value! ###
            ######################
            d = 2.353381 #mas, diameter of calibrator star 
            u = hdu['oi_vis'].data['ucoord']
            v = hdu['oi_vis'].data['vcoord']
            bls = np.sqrt( np.square(u) + np.square(v)  )
            wl_calib = df_calib_cflux['wlarr'][0]#/1e6

            
            #use uniform disk Fourier Transform to correct the visibilities
            for b in range(len(df_calib_cflux['corrflux'])):
                vis = np.absolute( 2 * jv(0, np.pi*d/1000/206265* bls[b]/wl_calib  ) / (np.pi*d/1000/206265* bls[b]/wl_calib)   )
                df_calib_cflux['corrflux'][b] /= vis
    
    
    for b in range(6):
        #correlated fluxes
        targ_flux = df_targ_cflux['corrflux'][b]
        targ_err = df_targ_cflux['corrfluxerr'][b]
        calib_flux = df_calib_cflux['corrflux'][b]
        calib_err = df_calib_cflux['corrfluxerr'][b]
        wls_calib = df_calib_cflux['wlarr'][0]
        
        calib_flux = np.interp(wls, wls_calib, calib_flux)
        calib_err = np.interp(wls, wls_calib, calib_err)

        #print(np.nanmean(targ_flux[s] - np.absolute(targ_err[s]) ))

        mask[b] = True #np.nanmean(targ_flux[s] - np.absolute(targ_err[s]) ) > 0
        calibrated_flux = targ_flux / calib_flux * true_flux
        relerr = np.sqrt( np.square(targ_err/targ_flux) + np.square(calib_err/calib_flux)  )   
        calibrated_err = calibrated_flux * relerr
        flux_vals[b] = calibrated_flux
        err_vals[b] = calibrated_err

        #do phases?

    for b in range(4):
        #closure phases
        targ_t3phi = df_targ_cphase['t3phi'][b]
        targ_t3err = df_targ_cphase['t3phierr'][b]

        calib_t3phi = df_calib_cphase['t3phi'][b]
        calib_t3err = df_calib_cphase['t3phierr'][b]
        #first subtract the calibrator phase
        new_t3phi = np.angle( np.exp(1j*(np.radians(targ_t3phi)-np.radians(calib_t3phi)) ), deg=True)
        #set errors on "long" triangles to 180
        new_t3phierr = np.copy(targ_t3err)
        if b in [2,3]:
            print('gets here')
            new_t3phierr = np.ones(targ_t3err.shape) * 360  
        #new_t3phierr[3] = np.ones(targ_t3err[3].shape) * 180
        #new_t3phierr[4] = np.ones(targ_t3err[3].shape) * 180
        t3phi_vals[b] = new_t3phi
        t3phi_errs[b] = new_t3phierr


    #tpl = targ_cflux.split('_')[-1].split('.pk')[0].split('T')[0] + 'T'+targ_cflux.split('_')[-1].split('.pk')[0].split('T')[1].replace('-',':')
    tpl = targ_tpl.split('T')[0] +'T'+ targ_tpl.split('T')[-1].replace('-',':')
    #cflux
    data_dict = {'corrflux':flux_vals, 'corrfluxerr':err_vals, 'mask':mask, 'tpl':[tpl]*6, 'wlarr':df_targ_cflux['wlarr']}
    df = pd.DataFrame.from_dict( data_dict )
    df.to_pickle('./lm_formatting/CALIBRATED_bcdcal_cflux_{tpl}.pk'.format(tpl=targ_tpl))

    #cphase
    data_dict = {'t3phi':t3phi_vals, 't3phierr':t3phi_errs, 'tpl':[tpl]*4, 'wlarr':df_targ_cphase['wlarr']}
    df = pd.DataFrame.from_dict( data_dict )
    df.to_pickle('./lm_formatting/CALIBRATED_bcdcal_t3phi_{tpl}.pk'.format(tpl=targ_tpl))


    if debug > 0:
        fig, axarr= plt.subplots(2,3)
        for b in range(len(flux_vals)):
            color = COLORS[b]
            if not mask[b]:
                color = 'grey'
            axarr.flatten()[b].errorbar(wls*1e6, flux_vals[b],yerr=err_vals[b], color=color)
            axarr.flatten()[b].set_xlabel('wavlength')
            axarr.flatten()[b].set_ylabel('flux [Jy]')
            axarr.flatten()[b].set_ylim([0,.1])
        plt.tight_layout()
        plt.savefig('./custom_pipeline/calibrated_cflux_{tpl}.png'.format(tpl=targ_tpl))
        plt.close()

        fig2, axarr= plt.subplots(2,2)
        for b in range(len(t3phi_vals)):
            color = COLORS[b]
            #if not mask[b]:
            #    color = 'grey'
            axarr.flatten()[b].errorbar(wls*1e6, t3phi_vals[b],yerr=t3phi_errs[b], color=color,alpha=0.75)
            axarr.flatten()[b].plot(wls*1e6, t3phi_vals[b],color='k')
            axarr.flatten()[b].set_xlabel('wavlength')
            axarr.flatten()[b].set_ylabel('closure phase [deg]')
            axarr.flatten()[b].set_ylim([-180,180])
        
        plt.tight_layout()
        plt.savefig('./custom_pipeline/calibrated_t3phi_{tpl}.png'.format(tpl=targ_tpl))
        plt.close()



def main():
    #each obs is a pair of strings (target tpl, calibrator tpl)
    observations = [        ('2020-03-13T04-02-11', '2020-03-13T04-40-24'),\
                            ('2020-03-13T04-56-22', '2020-03-13T04-40-24'),\
                        ('2020-03-14T03-53-00', '2020-03-14T05-59-29'),\
                        ('2020-03-14T04-31-58', '2020-03-14T05-59-29'),\
                        ('2020-03-14T04-51-12', '2020-03-14T05-59-29'),\
                        ('2020-03-14T07-57-12', '2020-03-14T08-31-10'),\
                        ('2020-03-14T08-57-48', '2020-03-14T08-31-10'),\
                            ('2021-02-28T06-32-19', '2021-02-28T07-07-46'),\
                            ('2021-02-28T07-42-00', '2021-02-28T07-07-46'),\
                            ('2021-02-28T07-22-31', '2021-02-28T07-07-46'),\
                        ('2021-06-01T03-10-17', '2021-06-01T03-59-25'),\
                        ('2021-06-01T04-29-41', '2021-06-01T03-59-25') 
                    ]

    for obs in observations:
        targ, cal = obs
        do_calibration(targ, cal, debug=1)

if __name__ == "__main__":
    main()