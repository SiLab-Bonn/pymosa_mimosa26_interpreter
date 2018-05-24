#!/usr/bin/env python
import sys,time,os
import testbeam_analysis.converter.pybar_fei4_converter as fe_cv
for i,p in enumerate(sys.path):
    if "pyBAR_mimosa26_interpreter" in p:
       del(sys.path[i])
       break
sys.path.append("../pyBAR_mimosa26_interpreter")
import simple_converter as m26_cv

debug=0
tr=False
ow=False
frame=False
if len(sys.argv)<2:
   print "ex_simple_converter.py [options] <input file>"
   print "options: -tr    transpose col and row of M26"
   print "         -ow    re-convert and overwrite output files"
   print "         -debug keep ntermidiate files"
   sys.exit()
if os.path.isdir(sys.argv[-1]):
    flist=[]
    flist_all=os.listdir(sys.argv[-1])
    for f in flist_all:
        if 'm26_telescope_scan.h5' in f:
            flist.append(os.path.join(sys.argv[-1],f))
else:
    flist=[sys.argv[-1]]
    flist_all=os.listdir(os.path.dirname(sys.argv[-1]))

if "-tr" in sys.argv[1:-1]:
    tr=True
if "-debug" in sys.argv[1:-1]:
    debug=1
if "-ow" in sys.argv[1:-1]:
    ow=True
if "-frame" in sys.argv[1:-1]:
    frame=True

for fin in flist:
    print "++++++++++++%s"%fin
    ### convert FE
    fe_tmp=fin[:-3]+"_event_aligned.h5"
    #if not os.path.basename(fe_tmp) in flist_all or ow:
    #    fe_cv.process_dut(fin)
    ### convert M26
    for i in range(1,7):  
        m26_tmp=os.path.join(fin[:-3]+"_event_aligned%d.h5"%i)
        if not os.path.basename(m26_tmp) in flist_all or ow:
            m26_cv.m26_converter(fin,m26_tmp,i)
        if frame==True:
            fout=fin[:-3]+"_frame_aligned%d.h5"%i
        else:
            fout=fin[:-3]+"_aligned%d.h5"%i
        if not os.path.basename(fout) in flist_all or ow:
            m26_cv.align_event_number(m26_tmp,fe_tmp,fout,tr=tr,frame=frame)
        if debug==0:
            try:
                os.remove(m26_tmp)
            except:
                pass
    if debug==0:
        try:
            os.remove(fe_tmp)
            os.remove(fin[:-3]+"_interpreted.h5")
        except:
            pass
