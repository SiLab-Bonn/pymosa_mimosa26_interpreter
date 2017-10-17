#!/usr/bin/env python
import sys,time,os
from numba import njit
import numpy as np
import tables

in_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u4'), 
                      ('x', '<u2'), ('y', '<u2'),('frame', '<u4')])
hit_dtype = np.dtype([('event_number', '<i8'), ('frame','u1'),
                      ('column', '<u2'), ('row', '<u2'),('charge', '<u2')])
@njit
def _convert(fe_hits,m26_hits,hits,factor_x,offset_x,factor_y,offset_y,tr,debug):
    hit_i=0
    m26_i=0
    for fe_i in range(fe_hits.shape[0]):
        fe_trig=fe_hits[fe_i]["trigger_number"] & 0x7FFF
        while m26_i < m26_hits.shape[0]:
           #print fe_i,m26_i,fe_hits[fe_i]["event_number"],m26_hits[m26_i]["event_number"],
           #print hex(fe_hits[fe_i]["trigger_number"]),hex(m26_hits[m26_i]["trigger_number"])
           m26_trig=m26_hits[m26_i]["trigger_number"] & 0x7FFF
           if (m26_trig-fe_trig) & 0x4000 == 0x4000:
               m26_i=m26_i+1
           elif m26_hits[m26_i]["trigger_number"]==fe_hits[fe_i]["trigger_number"]:
               hits[hit_i]["event_number"] = fe_hits[fe_i]["event_number"]
               if tr==True:
                   hits[hit_i]["row"] = offset_x + m26_hits[m26_i]["x"]*factor_x
                   hits[hit_i]["column"] = offset_y + m26_hits[m26_i]["y"]*factor_y
               else:
                   hits[hit_i]["row"] = offset_y + factor_y*m26_hits[m26_i]["y"]
                   hits[hit_i]["column"] = offset_y + factor_x*m26_hits[m26_i]["x"]

               hits[hit_i]['frame']= 0
               hits[hit_i]['charge']=1

               hit_i=hit_i+1
               m26_i=m26_i+1
           else:
               break
        if m26_i == m26_hits.shape[0]:
           return hits[:hit_i],fe_i,m26_i,0
    return hits[:hit_i],fe_i+1,m26_i,0

def convert_h5(fin,fe_fin,fout,factor_x=1,offset_x=1,factor_y=1,offset_y=1,tr=False,debug=0):
    n = 10000000
    debug=1
    buf = np.empty(n, dtype=hit_dtype)

    with tables.open_file(fe_fin) as fe_tb:
        fe_data = fe_tb.root.Hits[:][["event_number","trigger_number"]]
    fe_data,dummy=np.unique(fe_data,return_index=True)
    fe_end=len(fe_data)
    print "# of fe event",fe_end
    fe_start=0

    with tables.open_file(fin) as m26_tb:
      m26_end=int(len(m26_tb.root.Hits))
      print "m26",m26_end,
      m26_start=0
      with tables.open_file(fout,'w') as out_tb:
        description = np.zeros((1, ), dtype=hit_dtype).dtype
        hit_table = out_tb.create_table(out_tb.root, 
                    name='Hits', 
                    description=description, 
                    title='hit_data')

        t0 = time.time()
        while True:
           m26_tmpend=min(m26_start+n,m26_end)
           hits,fe_i,m26_i,err= _convert(fe_data[fe_start:], m26_tb.root.Hits[m26_start:m26_tmpend], buf,
                                factor_x,offset_x,factor_y,offset_y,tr,debug)
           t1=time.time()-t0
           if err==0:
               print fe_start,m26_start,fe_i,m26_i,err,"---%.3f%% %.3fs(%.3fus/dat)"%((m26_tmpend*100.0)/m26_end, t1, (t1)/m26_tmpend*1.0E6)
           else:
               print "ERROR=%d"%err,hit_i,fe_i,hex(fe_tb.root.Hits[fe_start+fe_i]["event_number"]),
               print m26_i,hex(m26_tb.root.Hits[m26_start+m26_i]["event_number"])
               m26_i=m26_i+1
               fe_i=fe_i+1
           hit_table.append(hits)
           hit_table.flush()
           fe_start=fe_start+fe_i
           m26_start=m26_start+m26_i
           if fe_start>=fe_end or m26_start>=m26_end:
               break

def convert_fe_h5(fin,fout,col_offset=0,col_factor=1,row_offset=0,row_factor=1,tr=True,debug=0):
    with tables.open_file(fin) as fe_tb:
        fe_data = fe_tb.root.Hits[:][["event_number","relative_BCID","column","row","tot"]]
    fe_data=fe_data[fe_data['row']!=0]
    print "# of hits in fe",len(fe_data)
    with tables.open_file(fout,'w') as out_tb:
        description = np.zeros((1, ), dtype=hit_dtype).dtype
        hit_table = out_tb.create_table(out_tb.root, 
                    name='Hits', 
                    description=description, 
                    title='hit_data')
        buf = np.empty(len(fe_data), dtype=hit_dtype)
        buf["event_number"]=fe_data["event_number"]
        buf["frame"]=fe_data["relative_BCID"]
        if tr:
            buf["column"]=row_offset+row_factor*fe_data["row"]
            buf["row"]=col_offset+col_factor*fe_data["column"]
        else:
            buf["column"]=col_offset+col_factor*fe_data["column"]
            buf["row"]=row_offset+row_factor*fe_data["row"]
        buf["charge"]=fe_data["tot"]
        hit_table.append(buf)
        hit_table.flush()


if __name__=="__main__":
    import os,sys
    debug=0

    if len(sys.argv)<2:
       print "simple_converter.py <input file> <ref file>"

    fin=sys.argv[-2]   ## fout of simple_event_builder
    fref=sys.argv[-1]  ## merged pyBAR_interpreted h5 file
      
    for i in range(1,7):
        print "------------------",i,"-----------------"
        fin_e=fin[:-3]+"_tlu%d.h5"%i
        fout=fin[:-3]+"_ev%d.h5"%i
        convert_h5(fin_e,fref,fout)
    print "------------------ fe -----------------"
    fout=fin[:-3]+"_ev.h5"
    convert_fe_h5(fref,fout)


