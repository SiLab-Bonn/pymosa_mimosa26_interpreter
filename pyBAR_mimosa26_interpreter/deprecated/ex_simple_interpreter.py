''' Example of simple_interpreter.py
'''

import sys,os,time
import tables
def img_size(w,h):
  import matplotlib
  pltsize=2
  matplotlib.rcParams["font.size"]=7*pltsize
  matplotlib.rcParams["legend.fontsize"]="small"
  matplotlib.rcParams['figure.figsize']=4*pltsize*w,3*pltsize*h
  matplotlib.rcParams["axes.color_cycle"]=["r","b","g","m","c","y","k","w"]

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from pyBAR_mimosa26_interpreter import simple_interpreter

@njit
def _mk_hist(hits,hist):
    for h in hits:
        if h["plane"]>0 and h["plane"]<=6:
            hist[h["plane"]-1,h["y"],h["x"]]=hist[h["plane"]-1,h["y"],h["x"]] + 1
    return hist

def mk_hist(fin,fout,noisy=800,limframe=-1):
    img_size(2,1.4)
    hist=np.zeros([6,576,1152])
    n=100000
    with tables.open_file(fin) as f:
        end=len(f.root.Hits)
        if limframe>0:
          end=min(end,limframe)
        start=0
        while start<end:
            tmpend=min(end,start+n)
            hits=f.root.Hits[start:tmpend]
            hist=_mk_hist(hits,hist)
            start=start+n

    vmax= np.sort(np.reshape(hist[2,:,:],1152*576))[-noisy]
    for i in range(6):
       plt.subplot(230+i+1)
       plt.imshow(hist[i,:,:],vmax=vmax,aspect="auto")
       plt.colorbar()

    plt.tight_layout()
    plt.savefig(fout)
    return fout
            
def mk_plot(fin, fout,limframe=10000):
    img_size(2,0.7)
    fig,ax=plt.subplots(1,3,sharey=True)
    n=100000
    last=-1
    with tables.open_file(fin) as f:
        total=len(f.root.Hits)
        cnt_all=[np.empty(0)]*7
        for i in range(total/n):
            end=min((i+1)*n,total)
            hits=f.root.Hits[i*n:end]
            for j in range(7):
                cnt=np.unique(hits['mframe'][hits['plane']==j],return_counts=True)
                if len(cnt[0])==0:
                    continue
                if last==cnt[0][0]:
                    cnt_all[j][-1]=cnt_all[j][-1]+cnt[1][0]
                    cnt_all[j]=np.append(cnt_all[j],cnt[1][1:])
                else:
                    cnt_all[j]=np.append(cnt_all[j],cnt[1])
                last=cnt[0][-1]
            if len(cnt_all[6])>limframe:
                break
    bins=np.arange(0,max(cnt_all[6])*1.25,1)
    binsfe=np.arange(0,max(cnt_all[0])*1.25,1)
    ax[0].hist(cnt_all[0],bins=binsfe,histtype="step",label="FEI4",normed=True);
    ax[1].hist(cnt_all[1],bins=bins,histtype="step",label="M26_1",normed=True);
    ax[1].hist(cnt_all[2],bins=bins,histtype="step",label="M26_2",normed=True);
    ax[1].hist(cnt_all[3],bins=bins,histtype="step",label="M26_3",normed=True);
    ax[2].hist(cnt_all[4],bins=bins,histtype="step",label="M26_4",normed=True);
    ax[2].hist(cnt_all[5],bins=bins,histtype="step",label="M26_5",normed=True);
    ax[2].hist(cnt_all[6],bins=bins,histtype="step",label="M26_6",normed=True);
    ax[1].set_ybound(0,10000)
    print "plane, average hits/frame (first %d frames)"%len(cnt_all[6])
    for j in range(7):
        print j,np.average(cnt_all[j])
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[0].set_xlabel("hits/frame")
    ax[0].set_ylabel("#")
    ax[1].set_xlabel("hits/frame")
    #ax[1].set_ylabel("#")
    ax[2].set_xlabel("hits/frame")
    #ax[2].set_ylabel("#")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(fout)
    return fout
if __name__=="__main__":
        fin=sys.argv[-1]
        fout=fin[:-3]+'.png'
        fout2=fin[:-3]+'2.png'
        fout_hit=fin[:-3]+'hits.h5'
        simple_interpreter.m26_interpreter(fin,fout_hit)
        print fout_hit
        print mk_plot(fout_hit,fout,limframe=10000)
        print mk_hist(fout_hit,fout2,noisy=800)
