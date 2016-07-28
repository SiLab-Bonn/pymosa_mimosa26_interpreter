''' Example of simple_interpreter.py
'''

import sys,os,time
import tables
import matplotlib
pltsize=2
matplotlib.rcParams["font.size"]=7*pltsize
matplotlib.rcParams["legend.fontsize"]="small"
matplotlib.rcParams['figure.figsize']=8*pltsize,2.1*pltsize

import matplotlib.pyplot as plt
import numpy as np

from pyBAR_mimosa26_interpreter import simple_interpreter

def mk_plot(fin, fout,limframe=10000):
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
    datdir='/sirrush/thirono/testbeam/2016-05-31/mimosa'
    distdir='/sirrush/silab/Toko/elsa_160531_hits_per_frame'
    flist=np.sort(os.listdir(datdir))
    donelist=os.listdir(distdir)
    for f in flist:
        if 'test_m26_telescope_scan.h5' not in f:
            continue
        if f[:-3]+'.png' in donelist:
            continue
        if f in ['18_test_m26_telescope_scan.h5']:
            continue
        fin=os.path.join(datdir,f)
        fout=os.path.join(distdir,f[:-3]+'.png')
        try:
            simple_interpreter.m26_interpreter(fin,"hits.h5")
        except:
            print "%s has error"%fin
        print fin
        try:
            mk_plot("hits.h5", fout,limframe=10000)
        except:
            print "%s has error during plotting"%fin
            pass
    os.remove("hits.h5")



