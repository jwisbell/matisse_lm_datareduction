import numpy as np 
from astropy.io import fits 
import matplotlib.pyplot as plt 
from glob import glob 
import sys
from sys import argv 
import pandas as pd 
import configparser


COLORS = ['firebrick','lightgreen','navy','mediumorchid','coral','cyan','k']

def load_data(fname):
    print(fname)
    hdu = fits.open(fname)
    
    time = hdu[1].data['time']
    real = hdu[1].data['corrfluxreal1']
    imag = hdu[1].data['corrfluximag1']
    tpl = hdu[0].header['eso tpl start']
    bcd = hdu[0].header['HIERARCH ESO CFG BCD MODE']
    print(bcd)

    print(real.shape, time.shape)
    return real, imag, time,tpl, bcd.lower()

def create_tracks():
    
    spectra = []
    f1spec = np.interp( np.linspace(110,0,110) ,[110,56,0][::-1] ,[27+3,31+3,39+3][::-1] )
    
    sky_spec = np.arange(0,110,1)
    #print(f1spec)
    #f2,f2cov = curve_fit(parabola,[45,52,80], [110,56,6],p0=[4,45])
    f2spec = np.interp(np.linspace(110,0,110),[110,56,6][::-1],[45,52,67][::-1]   )

    f3spec = np.interp(np.linspace(110,0,110),[110,39,6][::-1],[62,79,94][::-1] )

    f4spec = np.interp(np.linspace(110,0,110),[110,80,56,41,6][::-1],[80,85,95,100,122][::-1]    )

    f5spec = np.interp(np.linspace(110,0,110),[110,85,56,14,6][::-1],[95,102,116,143,157][::-1]    )

    f6spec = np.interp(np.linspace(110,0,110),[110,85,56,25,6][::-1],[110,120,134,159,183][::-1]    )

    skyspec = np.interp(np.linspace(110,0,110),[110,85,56,25,6][::-1],[250+12,250+12,250+12,250+12,250+12][::-1]    )

    spectra.append(f1spec)
    spectra.append(f2spec)
    spectra.append(f3spec)
    spectra.append(f4spec)
    spectra.append(f5spec)
    spectra.append(f6spec)
    spectra.append(skyspec)
    
    
    return spectra, sky_spec

def main(configfile, debug=1,bcd_combine=False,w = 3):
    print(configfile)
    cf = configparser.ConfigParser()
    cf.read(configfile)

    #0. parse config file
    target      = cf['Target']['target']
    rawdir      = str(cf['Data']['datadir'])
    wlmin       = float(cf['Data']['wlmin'])
    wlmax       = float(cf['Data']['wlmax'])
    opdstrict   = int(cf['Data']['opdstrict'])

    #0.5 define the fringe tracks for later
    #in principle the positions are always good because the shifts are corrected by the time they get to these files
    spectra_coords, sky_track  = create_tracks()


    #1. glob all the obj corr flux files
    length = len(glob(rawdir+'/OBJ_CORR_FLUX_*.fits')) #process them in order (cant use sort because it puts 10 before 2)
    print(rawdir)

    #2. for each file
    #   read in complex values
    #   flag individual frames
    #   extract complex fringes in each frame -- sort properly based on bcd 
    
    fringe_dict = {'out-out':{'t3t4':[],'t1t2':[],'t1t3':[],'t1t4':[],'t2t3':[],'t2t4':[], 'sky':[]},\
            'out-in':{'t3t4':[],'t1t2':[],'t1t3':[],'t1t4':[],'t2t3':[],'t2t4':[], 'sky':[]},\
            'in-out':{'t3t4':[],'t1t2':[],'t1t3':[],'t1t4':[],'t2t3':[],'t2t4':[], 'sky':[]},\
            'in-in':{'t3t4':[],'t1t2':[],'t1t3':[],'t1t4':[],'t2t3':[],'t2t4':[], 'sky':[]}  }
    for k in range(length):
        if k < 9:
            real, imag, time,tpl,bcd_config = load_data(rawdir+'/OBJ_CORR_FLUX_000%i.fits'%(k+1))
        else:
            real, imag, time,tpl,bcd_config = load_data(rawdir+'/OBJ_CORR_FLUX_00%i.fits'%(k+1))
        
        complex_data = real + 1j*imag

        #here we can do flagging based on opds
        #comment out if you want to use all frames
        tpl_temp = tpl.replace(":","_")
        try:
            x = np.load('./opd_flagging/%s_000%i.npy'%(tpl_temp,k+1))
        except:
            x = np.load('./opd_flagging/%s_00%i.npy'%(tpl_temp,k+1))
        
        my_mjds = x[-1]
        n_flagged = [0 for _ in range(len(my_mjds))]
        colors = ['cyan' for _ in range(len(my_mjds))]
        n_perf = 0
        n_good = 0
        n_okay = 0
        for i in range(len(my_mjds)):
            n_flagged[i] = int(x[0,i]) + int(x[1,i]) + int(x[2,i]) + int(x[3,i]) + int(x[4,i]) + int(x[5,i])
            if n_flagged[i] <= 1:
                colors[i] = 'forestgreen'
                if n_flagged[i] == 0:
                    n_perf += 1
                n_good += 1
                n_okay += 1
            elif n_flagged[i] == 2:
                n_okay += 1
                colors[i] == 'green'
            elif n_flagged[i] == 3:
                colors[i] = 'yellow'
            else:
                colors[i] = 'firebrick'
        n_flagged = np.array(n_flagged)
        
        #compare to the opdstrict param
        mask = n_flagged <= opdstrict
        if debug > 1:
            fig = plt.figure()
            plt.title('Percent = 0:  %.1f     Percent <=1:  %.1f       Percent <=2:  %.1f'%( n_perf / len(my_mjds) * 100,  n_good / len(my_mjds) * 100 , n_okay/len(my_mjds)*100   ))
            plt.bar(my_mjds, n_flagged, width = np.diff(my_mjds)[0],color=colors )
            plt.show()
            plt.close()

        #apply frame mask    
        complex_data = complex_data[mask,:,310:]
        

        complex_fringes = [[],[],[],[],[],[], []] #7th is bias estimate
        #extract complex fringes from each unmasked frame 
        
        for j in range(len(complex_data)):
            data = np.flipud(complex_data[j])
            for l in [0,1,2,3,4,5,6]:#range(len(spectra_coords)):
                tempx = spectra_coords[l] - 12
                tempy = np.linspace(110,0,len(tempx))
                
                targ = np.zeros(data.shape, dtype=np.complex)
                targ[targ<1e-3] = np.nan
                
                
                for i in range(len(tempx)):
                    #print(data[ int(tempy[i])-1:int(tempy[i])+1, int(tempx[i])-w:int(tempx[i])+w ],'one')
                    #print(targ[ int(tempy[i])-1:int(tempy[i])+1, int(tempx[i])-w:int(tempx[i])+w ],'two')
                    targ[ int(tempy[i])-1:int(tempy[i])+1, int(tempx[i])-w:int(tempx[i])+w ] = data[ int(tempy[i])-1:int(tempy[i])+1, int(tempx[i])-w:int(tempx[i])+w ] #- bias_targ
                    

                y = np.array(np.nanmean(targ,1)[::1])
                complex_fringes[l].append( y )

        wl_arr = np.linspace(wlmin,wlmax, complex_data[0].shape[0] )
        if debug > 0:
            im = np.flipud(np.absolute(np.nanmean(complex_data,0)))
            fig,axarr = plt.subplots(1,3,figsize=(12,4))
            axarr[0].imshow(np.log10(im), origin='lower', vmax=1)
            
            for l in [0,1,2,3,4,5,6]:
                tempx = spectra_coords[l] - 12
                tempy = np.linspace(110,0,len(tempx))
                #axarr[0].plot(tempx, tempy,color=COLORS[l])
                #axarr[0].plot(tempx-w, tempy,ls='--',color=COLORS[l])
                #axarr[0].plot(tempx+w, tempy,ls='--',color=COLORS[l])
                
                
                axarr[1].errorbar( wl_arr, np.nanmean( np.absolute(complex_fringes[l]),0 ),yerr=np.nanstd( np.absolute(complex_fringes[l]),0),color=COLORS[l]   )
                axarr[2].errorbar( wl_arr, np.angle( np.nanmean(complex_fringes[l],0) ),yerr=np.nanstd( np.angle(complex_fringes[l]),0),color=COLORS[l]   )
            plt.suptitle(target + ' ' + tpl_temp )
            #plt.show()
            plt.tight_layout()
            plt.savefig('./custom_pipeline/{tpl}_fringes.png'.format(tpl=tpl_temp))
            plt.close()
            
        #need to sort the fringes based on the current bcd configuration to make them consistent later
        if bcd_combine:
            if bcd_config == 'out-out':
                fringe_dict['out-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-out']['t2t3'] += complex_fringes[2]  
                fringe_dict['out-out']['t2t4'] += complex_fringes[3]  
                fringe_dict['out-out']['t1t3'] += complex_fringes[4]  
                fringe_dict['out-out']['t1t4'] += complex_fringes[5]  
                fringe_dict['out-out']['sky']  += complex_fringes[6]  
            elif bcd_config == 'out-in':
                fringe_dict['out-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-out']['t1t3'] += complex_fringes[2]  
                fringe_dict['out-out']['t1t4'] += complex_fringes[3]  
                fringe_dict['out-out']['t2t3'] += complex_fringes[4]  
                fringe_dict['out-out']['t2t4'] += complex_fringes[5]  
                fringe_dict['out-out']['sky']  += complex_fringes[6]  
            elif bcd_config == 'in-out':       
                fringe_dict['out-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-out']['t2t4'] += complex_fringes[2]  
                fringe_dict['out-out']['t2t3'] += complex_fringes[3]  
                fringe_dict['out-out']['t1t4'] += complex_fringes[4]  
                fringe_dict['out-out']['t1t3'] += complex_fringes[5]  
                fringe_dict['out-out']['sky']  += complex_fringes[6]  
            elif bcd_config == 'in-in':
                fringe_dict['out-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-out']['t1t4'] += complex_fringes[2]  
                fringe_dict['out-out']['t1t3'] += complex_fringes[3]  
                fringe_dict['out-out']['t2t4'] += complex_fringes[4]  
                fringe_dict['out-out']['t2t3'] += complex_fringes[5]  
                fringe_dict['out-out']['sky']  += complex_fringes[6]  
        else:
            if bcd_config == 'out-out':
                fringe_dict['out-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-out']['t2t3'] += complex_fringes[2]  
                fringe_dict['out-out']['t2t4'] += complex_fringes[3]  
                fringe_dict['out-out']['t1t3'] += complex_fringes[4]  
                fringe_dict['out-out']['t1t4'] += complex_fringes[5]  
                fringe_dict['out-out']['sky']  += complex_fringes[6]  
            elif bcd_config == 'out-in':
                fringe_dict['out-in']['t3t4'] += complex_fringes[0]  
                fringe_dict['out-in']['t1t2'] += complex_fringes[1]  
                fringe_dict['out-in']['t1t3'] += complex_fringes[2]  
                fringe_dict['out-in']['t1t4'] += complex_fringes[3]  
                fringe_dict['out-in']['t2t3'] += complex_fringes[4]  
                fringe_dict['out-in']['t2t4'] += complex_fringes[5]  
                fringe_dict['out-in']['sky']  += complex_fringes[6]  
            elif bcd_config == 'in-out':       
                fringe_dict['in-out']['t3t4'] += complex_fringes[0]  
                fringe_dict['in-out']['t1t2'] += complex_fringes[1]  
                fringe_dict['in-out']['t2t4'] += complex_fringes[2]  
                fringe_dict['in-out']['t2t3'] += complex_fringes[3]  
                fringe_dict['in-out']['t1t4'] += complex_fringes[4]  
                fringe_dict['in-out']['t1t3'] += complex_fringes[5]  
                fringe_dict['in-out']['sky']  += complex_fringes[6]  
            elif bcd_config == 'in-in':
                fringe_dict['in-in']['t3t4'] += complex_fringes[0]  
                fringe_dict['in-in']['t1t2'] += complex_fringes[1]  
                fringe_dict['in-in']['t1t4'] += complex_fringes[2]  
                fringe_dict['in-in']['t1t3'] += complex_fringes[3]  
                fringe_dict['in-in']['t2t4'] += complex_fringes[4]  
                fringe_dict['in-in']['t2t3'] += complex_fringes[5]  
                fringe_dict['in-in']['sky']  += complex_fringes[6] 
        
    

    ##make them into long arrays
    for bcd in ['out-out','out-in','in-out','in-in']:
        for key in fringe_dict[bcd].keys():
            fringe_dict[bcd][key] = np.array(fringe_dict[bcd][key])

    

    #3. process the complex fringes
    #   compute triple products for each 
    loops = [['t2t3','t3t4','t2t4'],['t1t2','t2t3','t1t3'],['t1t2','t2t4','t1t4'],['t1t3','t3t4','t1t4']   ]
    
    extracted_dict = {  'out-out': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                        'out-in': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                        'in-out': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                        'in-in': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] }}
    for bcd in ['out-out','out-in','in-out','in-in']:
        for index, loop in enumerate(loops):
            fringe1 = fringe_dict[bcd][loop[0]]
            fringe2 = fringe_dict[bcd][loop[1]]
            fringe3 = fringe_dict[bcd][loop[2]]

            #take sign flips due to bcds into account
            if bcd in ['out-in','in-in'] and index in [1,2]:
                fringe1 = np.conjugate(fringe1)

            if bcd in ['in-out','in-in'] and index in [0,3]:
                fringe2 = np.conjugate(fringe2)

            t3 = np.array(fringe1 * fringe2 * np.conjugate(fringe3))
            extracted_dict[bcd]['t3'][index] = t3

  
    #   compute raw corr fluxes for each 
    for bcd in ['out-out','out-in','in-out','in-in']:
        for l,key in enumerate(['t3t4','t1t2','t2t3','t2t4','t1t3','t1t4']):
            bias = np.absolute(fringe_dict[bcd]['sky'])
            raw_flux = np.absolute(fringe_dict[bcd][key])
            flux = raw_flux - bias 
            extracted_dict[bcd]['cflux'][l] = flux 

    #   compute raw phases for each 
    for bcd in ['out-out','out-in','in-out','in-in']:
        for l,key in enumerate(['t3t4','t1t2','t2t3','t2t4','t1t3','t1t4']):
            phase = np.angle(fringe_dict[bcd][key] - 0*fringe_dict[bcd]['sky'], deg=False)
            extracted_dict[bcd]['phase'][l] = phase
    
    
    if debug > 0:
        fig,axarr = plt.subplots(2,3)
        plt.suptitle(target + ' ' + tpl_temp)
        final_vals = [[],[],[],[],[],[]]
        for bcd in ['out-out','out-in','in-out','in-in']:
            for l,key in enumerate(['t3t4','t1t2','t2t3','t2t4','t1t3','t1t4']):
                
                sky = np.absolute(fringe_dict[bcd]['sky'])
                axarr.flatten()[l].errorbar(wl_arr*1e6, np.nanmedian(np.absolute(fringe_dict[bcd][key])-sky,0), yerr=0*np.nanstd(np.absolute(fringe_dict[bcd][key])-sky,0),alpha=0.5,color=COLORS[l] )
                axarr.flatten()[l].errorbar(wl_arr*1e6, np.nanmedian(extracted_dict[bcd]['cflux'][l],0), color='red'  )
                axarr.flatten()[l].set_ylim([-1,None])
                axarr.flatten()[l].set_title(key)
                axarr.flatten()[l].set_ylabel('flux [cts]')
                axarr.flatten()[l].set_xlabel('wl')
                final_vals[l].append(np.nanmedian(np.absolute(fringe_dict[bcd][key])-sky,0))
        for b in range(6):
            axarr.flatten()[b].errorbar(wl_arr*1e6, np.nanmedian(final_vals[b],0), yerr=np.nanstd(final_vals[b],0),color='k' )
        #plt.show()
        plt.tight_layout()
        plt.savefig('./custom_pipeline/{tpl}_cflux.png'.format(tpl=tpl_temp))
        plt.close()
        

        fig,axarr = plt.subplots(2,3)
        plt.suptitle(target + ' ' + tpl_temp)
        final_vals = [[],[],[],[],[],[]]
        for bcd in ['out-out','out-in','in-out','in-in']:
            for l,key in enumerate(['t3t4','t1t2','t2t3','t2t4','t1t3','t1t4']):                
                sky = np.absolute(fringe_dict[bcd]['sky'])
                axarr.flatten()[l].errorbar(wl_arr*1e6, np.angle(np.nanmean(fringe_dict[bcd][key],0),deg=True), yerr=0*np.nanstd(np.angle(fringe_dict[bcd][key],deg=True),0),alpha=0.5,color=COLORS[l] )
                axarr.flatten()[l].set_ylim([-180,180])
                axarr.flatten()[l].errorbar(wl_arr*1e6, np.angle(np.nanmean( np.exp(1j*extracted_dict[bcd]['phase'][l] ),0),deg=True), color='red'  )
                final_vals[l].append(np.nanmean(fringe_dict[bcd][key],0))
                axarr.flatten()[l].set_ylabel('phase [deg]')
                axarr.flatten()[l].set_xlabel('wl')
        for b in range(6):
            axarr.flatten()[b].errorbar(wl_arr*1e6, np.angle(np.nanmedian(final_vals[b],0),deg=True), yerr=np.nanstd(np.angle(final_vals[b],deg=True),0),color='k' )
        #plt.show()
        plt.tight_layout()
        plt.savefig('./custom_pipeline/{tpl}_phase.png'.format(tpl=tpl_temp))
        plt.close()
    
    #4. take means where necessary
    mean_dict = {   'out-out': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                    'out-in': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                    'in-out': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] },\
                    'in-in': {'t3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] }}

    final_vals = { 't3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] }
    final_errs = { 't3':[[],[],[],[]],'cflux':[[],[],[],[],[],[]],'phase':[[],[],[],[],[],[]] }
    #first take mean within each bcd for each value, then do bcd averaging

    for bcd in ['out-out','out-in','in-out','in-in']:
        key = 'cflux'
        for l in range(6):
            mean_dict[bcd][key][l] = np.nanmedian(extracted_dict[bcd]['cflux'][l],0)

        key = 'phase'
        for l in range(6):
            mean_dict[bcd][key][l] = np.angle(np.nanmean(  np.exp( 1j* extracted_dict[bcd][key][l]) ,0) )

        key = 't3'
        for l in range(4):
            mean_dict[bcd][key][l] = np.angle(np.nanmean( extracted_dict[bcd][key][l],0) )

    #now bcd average
    for b in range(6):
        key = 'cflux'
        final_vals[key][b] = np.nanmean( [ mean_dict[bcd][key][b] for bcd in ['out-out','out-in','in-out','in-in'] ],0    )
        final_errs[key][b] = np.nanstd( [ mean_dict[bcd][key][b] for bcd in ['out-out','out-in','in-out','in-in'] ],0    )

        key = 'phase'
        final_vals[key][b] = np.angle(np.nanmean( [ np.exp(1j*mean_dict[bcd][key][b]) for bcd in ['out-out','out-in','in-out','in-in'] ],0),deg=True)
        final_errs[key][b] = np.nanstd(np.angle( [ np.exp(1j*mean_dict[bcd][key][b]) for bcd in ['out-out','out-in','in-out','in-in'] ],deg=True),0)
    
    for b in range(4):
        key = 't3'
        final_vals[key][b] = np.angle(np.nanmean( [ np.exp(1j*mean_dict[bcd][key][b]) for bcd in ['out-out','out-in','in-out','in-in'] ],0),deg=True)
        final_errs[key][b] = np.nanstd(np.angle( [ np.exp(1j*mean_dict[bcd][key][b]) for bcd in ['out-out','out-in','in-out','in-in'] ],deg=True),0)

    if debug > 0:
        fig,axarr = plt.subplots(2,2)
        labels = ['2-3-4','1-2-3','1-2-4','1-3-4']
        
        for b in range(4):
            for bcd in ['out-out','out-in','in-out','in-in']:
                axarr.flatten()[b].errorbar(wl_arr*1e6, np.degrees(mean_dict[bcd]['t3'][b]),alpha=0.5  )

            axarr.flatten()[b].errorbar(wl_arr*1e6, final_vals['t3'][b],yerr=final_errs['t3'][b], color='k' )
            axarr.flatten()[b].set_ylim([-180,180])
            axarr.flatten()[b].set_title(labels[b])
            axarr.flatten()[b].set_ylabel('cphase [deg]')
            axarr.flatten()[b].set_xlabel('wl')
        plt.suptitle(target + ' ' + tpl_temp)
        #plt.show()
        plt.tight_layout()
        plt.savefig('./custom_pipeline/{tpl}_cphase.png'.format(tpl=tpl_temp))
        plt.close()







    #5. format for saving and write to disk
    #to do
    
    cflux_dict = {'corrflux':final_vals['cflux'], 'corrfluxerr':final_errs['cflux'],\
         'wlarr':[wl_arr]*6,'bl_name':['t3t4','t1t2','t2t3','t2t4','t1t3','t1t4']\
        ,'rawphase':final_vals['phase'],'rawphaseerr':final_errs['phase']}
    df = pd.DataFrame.from_dict( cflux_dict )
    df.to_pickle('./custom_pipeline/bcd_corrflux_{tpl}.pk'.format(tpl=tpl.replace(':','-')))

    cphase_dict = {'t3phi':final_vals['t3'], 't3phierr':final_errs['t3'], 'wlarr':[wl_arr]*4,'triangle':['2-3-4','1-2-3','1-2-4','1-3-4']}
    df2 = pd.DataFrame.from_dict( cphase_dict )
    df2.to_pickle('./custom_pipeline/bcd_cphase_{tpl}.pk'.format(tpl=tpl.replace(':','-')))


    #bcd_dict = {'corrflux': bcd_vals, 'corrfluxerr':bcd_errs, 'bls':[[34,35],[32,33],[33,34],[33,35],[32,34],[32,35]],'wlarr':[data_dict['wlarr'][0]]*6  }
    
    #df.to_pickle('./bcd_averaged_{tpl}.pk'.format(tpl=tpl.replace(':','-')))
    
    






if __name__ =="__main__":
    if len(argv) < 2:
        print('ERROR!')
        print('Usage: python lm_cflux_cphase_extraction.py PATH/TO/CONFIGFILE')
        sys.exit()
    
    script, configfile = argv 
    main(configfile, debug=1)
