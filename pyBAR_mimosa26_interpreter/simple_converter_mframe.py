#!/usr/bin/env python
import sys,time,os
from numba import njit
import numpy as np
import tables

hit_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u2'),('event_timestamp','<u2'),('frame_timestamp','<u4'), 
                      ('frame','<u4'),('x', '<u2'), ('y', '<u2')])
tmp_dtype = np.dtype([("event_number","<i8"),("event_timestamp","<u8"),("column","<u2"),("row","<u2")])
m26_dtype = np.dtype([("event_number","<i8"),("frame","<u1"),("column","<u2"),("row","<u2"),("charge","<u2")])
buf_dtype = np.dtype([("event_number","<i8"),("event_timestamp","<u8"),("frame","<u1"),("column","<u2"),("row","<u2"),("charge","<u2")])

def convert_mframe_m26_h5(fins, fout,plane,x_offset=1,x_factor=1,y_offset=1,y_factor=1,tr=False,n=10000000,debug=0):
    if isinstance(fins,str):
        fins=[fins]
    with tables.open_file(fout, "w") as f_o:
      if debug & 0x1 ==0x1:
          description=np.zeros((1,),dtype=tmp_dtype).dtype
          tmp_table=f_o.create_table(f_o.root,name="TmpHits",description=description,title='tmphit_data')
      description=np.zeros((1,),dtype=m26_dtype).dtype
      hit_table=f_o.create_table(f_o.root,name="Hits",description=description,title='hit_data')
      buf=np.empty(n,dtype=buf_dtype)
      t0=time.time()
      total_data=0
      pre_event_number=-1
      for f_i,fin in enumerate(fins):
        with tables.open_file(fin) as f:
            start=0
            end=len(f.root.Hits)
            print "convert_mframe_m26_h5() # of plane%d=%d"%(plane,end)
            while start < end:
                tmpend=min(end,start+n)
                hit=f.root.Hits[start:tmpend]
                hit=hit[hit['plane']==plane][['timestamp','x','y','mframe']]
                #for i in hit[:10]:
                #    print i
                if len(hit)==0:
                    start=tmpend
                    continue
                if f_i==0 and pre_event_number==-1:
                    pre_timestamp=np.uint64(hit[0]["timestamp"])
                    pre_event_number=np.uint64(hit[0]["mframe"])
                #print pre_timestamp,pre_event_number
                tmp=np.uint64(hit["timestamp"])
                buf[:len(hit)]["event_timestamp"]=_fix_timestamp(tmp,pre_timestamp,overflow=np.uint64(0x100000000))
                #print buf[:10]["event_timestamp"]
                tmp=np.uint64(hit["mframe"])
                if np.all(hit["mframe"][1:]-hit["mframe"][:-1] >= 0):
                    buf[:len(hit)]["event_number"]=tmp
                else:
                    print "fix mframe",pre_event_number
                    buf[:len(hit)]["event_number"]=np.int64(_fix_timestamp(tmp,pre_event_number,overflow=np.uint64(0x100000000)))
                if debug & 0x1 == 0x1:
                    buf[:len(hit)]['row']=hit['x']
                    buf[:len(hit)]['column']=hit['y']
                    #print buf[:10]["event_number"]
                    tmp_table.append(buf[:len(hit)][["event_number","event_timestamp","column","row"]])
                    tmp_table.flush()
                if tr==True:
                    buf[:len(hit)]['row']=x_offset+x_factor*hit['x']
                    buf[:len(hit)]['column']=y_offset+y_factor*hit['y']
                else:
                    buf[:len(hit)]['column']=x_offset+x_factor*hit['x']
                    buf[:len(hit)]['row']=y_offset+y_factor*hit['y']

                buf[:len(hit)]['frame']=np.zeros(len(hit),dtype=np.uint8)
                buf[:len(hit)]['charge']=np.ones(len(hit),dtype=np.uint16)
                hit_table.append(buf[:len(hit)][hit["x"]<1152][["event_number","frame","column","row","charge"]])
                hit_table.flush()
                print "convert_mframe_m26_h5()","%.3f%% done"%(100.0*tmpend/end),"%.3fs"%(time.time()-t0)
                start=tmpend
                #break
    print "convert_mframe_m26_h5() DONE fout %s"%fout                
@njit
def _fix_timestamp(dat,pre_timestamp,overflow=np.uint64(0x8000)):
    mask=overflow-np.uint64(1)
    not_mask=~mask
    half_mask = mask >> np.uint64(1)
    #print hex(mask),hex(not_mask),hex(half_mask)
    for d_i,d in enumerate(dat):
        d= (mask  & d) | (not_mask  & pre_timestamp)
        #print d_i, hex(d), hex(pre_timestamp),hex(d +half_mask), d > pre_timestamp + half_mask,d +half_mask < pre_timestamp,
        if d > pre_timestamp + half_mask:
            #print "sub"
            d=d - overflow
        elif d +half_mask < pre_timestamp:
            #print "add"
            d=d + overflow

        else:
            #print "keep"
            pass
        dat[d_i]=d
        pre_timestamp=d
    return dat

if __name__=="__main__":
    import os,sys

    if len(sys.argv)<2:
       print "simple_converter_mframe.py <hit file> [hit file2].."
    fins=[]
    for f in sys.argv[1:]:
       fins.append(f[:-3]+"_hit.h5")
    fin=sys.argv[1]

    for i in range(1,7):
        print "------------",i,"---------------"
        fout=fin[:-3]+"_mframe%d.h5"%i
        print fout

        convert_mframe_m26_h5(fins,fout,i,debug=0)
        #break
