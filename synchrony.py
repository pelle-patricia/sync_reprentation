import numpy as np
from scipy import signal as sig
from math import *


def hn_cochlear(lh = 2048,
                Fs = 16000,
                df = 25./np.log(np.sqrt(2.)),
                fz = 2000.,
                fp_minus_fz = 10.*np.log(2.)/np.log(np.sqrt(2.))):
    """
        hn_cochlear Cochlear filter in discrete form. Parameters are in continuous freq form. 
        lh: filter order
        df: filter bandwidth
        fz: Cutoff filter frequency 
        fp_minus_fz: frequency distance from peak to zero frequency 
    """

    b = 1/(2.*np.pi*df)
    a = fp_minus_fz*2*np.pi*b
    om = np.linspace(0,Fs,int(lh),endpoint=False)*2*np.pi

    Hd = np.zeros(int(lh))
    Hd[om < fz*2.*np.pi] = (2*np.pi*fz - om[om < fz*2.*np.pi])**a*\
        np.exp(-b*(2*np.pi*fz - om[om < fz*2.*np.pi]))
    Hd = Hd+ Hd[-1::-1]
    tmp = np.fft.ifft(Hd)
    hd = np.fft.fftshift(tmp)

    return( hd )

class bank_pll:
    
    """    
        pll initialization and execution 
        This class implements an array of plls, to be calculated over a section of time.
        Output variables are numpy arrays of dimenension nplls by ltime.
        
        sita, Fn, Fo: parameters for internal loop in countinuos form (see tuning_curves.ipynb for explanation)
        fc_lock: Lockin filter freq cutoff.
        Fs: sampling freq. 
    """
        
    def __init__(self, nplls = 1, ltime=16000, sita=0.7, Fn=200., Fo=1975., fc_lock=4, Fs=16000.):

        self.n_f = 0
        self.N = ltime
        self.Fs = Fs
        #Following variables were being calculated in PLL
        self.ud = np.zeros((nplls,ltime))
        self.ul = np.zeros((nplls,ltime))
        self.ud_int = np.zeros(nplls)
        self.uf = np.zeros((nplls,ltime))
        self.uf_int = np.zeros(nplls)
        self.theta2 = np.zeros((nplls,ltime))
        self.vco = np.zeros((nplls,ltime))
        self.vco90 = np.zeros((nplls,ltime))
        self.vco90[:,-1] = np.ones(nplls)
        self.lock = np.zeros((nplls,ltime))
        self.agc = np.zeros((nplls,ltime))
        self.agc[:,-1] = np.ones(nplls)
        self.time2kpi = np.zeros((nplls,ltime))
        self.freq = np.zeros((nplls,ltime))
        self.nper = np.zeros(nplls)
        self.xprev = np.zeros(nplls)
        
        #Parameters:
        wn = 2*np.pi*Fn/Fs
        self.wo = 2*np.pi*Fo/Fs
        self.G0 = 1.
        self.G1 = (1 - np.exp(-2*sita*wn))/self.G0
        self.G2 = (1 + np.exp(-2*sita*wn) - 2.*np.exp(-sita*wn)*np.cos(wn*np.sqrt(1-sita*sita)))/self.G0
        [b,a] = sig.ellip(N=1,rp=1.01,rs=20,Wn=2*pi*fc_lock,analog=True)
        [self.block,self.alock] = sig.bilinear(b,a,Fs)
        
        
    def section_calc(self,x):
        
        uf_f = self.uf[:,-1]
        ud_f = self.ud[:,-1]
        theta2_f = self.theta2[:,-1]
        
        for n in range(self.N):
        # 1) u_d
            self.ud[:,n] = self.agc[:,n-1]*x[:,n]*self.vco[:,n-1]
            self.ul[:,n] = self.agc[:,n-1]*x[:,n]*self.vco90[:,n-1]
#             self.ud[:,n] = x[:,n]*self.vco[:,n-1]
#             self.ul[:,n] = x[:,n]*self.vco90[:,n-1]
        # 2) u_f:
            self.ud_int += self.ud[:,n]
            self.uf[:,n] = self.G1*(self.ud[:,n]-ud_f[:]) + self.G2*self.ud_int[:] + uf_f[:] 
        # 3) theta2 and vco:
            self.uf_int += self.uf[:,n]
            self.theta2[:,n] = self.G0*self.uf_int[:] + self.wo*(self.n_f+n) + theta2_f[:]
            self.vco[:,n] = np.cos(self.theta2[:,n])
            self.vco90[:,n] = np.cos(self.theta2[:,n] -np.pi/2.)
        # 4) AGC and lock-in calculation: first order IIR filter
            self.lock[:,n] = self.block[0]*self.ul[:,n] + self.block[1]*self.ul[:,n-1] - self.alock[1]*self.lock[:,n-1]
            self.agc[:,n] = 10./np.exp(np.abs(3*np.arctan(0.7*np.abs(self.lock[:,n]))))
        # 5) Freq estimation
            for k in range(self.ud.shape[0]):

                if( (np.floor(np.abs(self.theta2[k,n]/(2*np.pi))) != self.nper[k]) ):
                    self.nper[k] = np.floor(np.abs(self.theta2[k,n])/(2*np.pi))
                    self.time2kpi[k,n] = 1./self.Fs*(self.n_f + n-1 + \
                        np.abs(self.theta2[k,n-1]- 2*np.pi*self.nper[k])/\
                        (np.abs(self.theta2[k,n] - 2*np.pi*self.nper[k])+np.abs(self.theta2[k,n-1]-2*np.pi*self.nper[k])))
                        #last two lines are for better calculation of freq, computing a fraction of Ts interpolating linealy 
                    self.freq[k,n] = 1./(self.time2kpi[k,n] - self.time2kpi[k,n-1])
                else: 
                    self.time2kpi[k,n] = self.time2kpi[k,n-1]
                    self.freq[k,n] = self.freq[k,n-1]

