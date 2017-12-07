#!/usr/bin/env python
import sys,time,os
from numba import njit
import numpy as np
import tables

hit_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u2'),('event_timestamp','<u2'),('frame_timestamp','<u4'), 
                      ('frame','<u4'),('x', '<u2'), ('y', '<u2')])
m26_dtype=[("event_number","<u8"),("event_timestamp","<u8"),("trigger_number","<u2"),
           ("mframe_timestamp","<u8"),("mframe","<u4"),("x","<u2"),("y","<u2")]

@njit
def _build(m26, tlu, buf,event_number):
    d1_idx=0
    d2_idx=0
    buf_i=0
    while d1_idx < len(m26) and d2_idx < len(tlu) and buf_i<len(buf):
        #print d1_idx,d2_idx,m26[d1_idx]["mframe"],tlu[d2_idx]["mframe"]
        if m26[d1_idx]["mframe"] == tlu[d2_idx]["mframe"]+1 or m26[d1_idx]["mframe"] == tlu[d2_idx]["mframe"]+2:
            cnt=-1
            for i in range(d1_idx,len(m26)):
                if m26[i]["mframe"] > tlu[d2_idx]["mframe"]+2:
                    #print "break",i,d2_idx,m26[i]["mframe"]
                    cnt=i
                    break
            if cnt==-1:
                return 1, buf[:buf_i],d1_idx,d2_idx,event_number
            for i in range(d1_idx,cnt):
                buf[buf_i]["mframe"]=m26[i]["mframe"]
                buf[buf_i]["mframe_timestamp"]=m26[i]["timestamp"]
                buf[buf_i]["x"]=m26[i]["x"]
                buf[buf_i]["y"]=m26[i]["y"]
                buf[buf_i]["event_number"]=tlu[d2_idx]["mframe"] #event_number
                buf[buf_i]["event_timestamp"]=tlu[d2_idx]["timestamp"]
                buf[buf_i]["trigger_number"]=tlu[d2_idx]["tlu"]
                buf_i=buf_i+1
            d2_idx=d2_idx+1
        elif m26[d1_idx]["mframe"] > tlu[d2_idx]["mframe"]+2:
            #print "next2"
            d2_idx=d2_idx+1
            event_number=event_number+1
        elif m26[d1_idx]["mframe"] < tlu[d2_idx]["mframe"]+1:
            #print "next1"
            d1_idx=d1_idx+1
    return 0, buf[:buf_i],d1_idx,d2_idx,event_number

def build_h5(fins,fout,planes=[1,2,3,4,5,6],n=10000000):
    #### get trigger
    if isinstance(fins,str):
        fins=[fins]
    if isinstance(planes,int):
        planes=[planes]
    for f_i,fin in enumerate(fins):
        with tables.open_file(fin) as f:
            start=0
            end=len(f.root.Hits)
            print "# of data in %s"%fin,end
            while start < end:
                tmpend=min(end,start+n)
                tmp=f.root.Hits[start:tmpend]
                tmp=tmp[tmp['plane']==255]
                if start==0 and f_i==0:
                    tlu=tmp[['mframe','tlu','timestamp']]
                else:
                    tlu=np.append(tlu, tmp[['mframe','tlu','timestamp']])
                start=tmpend
                print ".",
            print 
    print "# of trigger",len(tlu)
    
    buf=np.empty(n,dtype=m26_dtype)

    for plane in planes:
        if len(planes)>1:
            fname_o=fout[:-3]+"%d.h5"%plane
        else:
            fname_o=fout
        with tables.open_file(fname_o, "w") as f_o:
          description=np.zeros((1,),dtype=m26_dtype).dtype
          hit_table=f_o.create_table(f_o.root,name="Hits",description=description,title='hit_data')
          t0=time.time()
          total_data=0
          event_number=0
          tlu_idx=0
          for f_i,fin in enumerate(fins):
            with tables.open_file(fin) as f:
                start=0
                end=len(f.root.Hits)
                while start < end:
                    tmpend=min(end,start+n)
                    tmp=f.root.Hits[start:tmpend]
                    if f_i==0 and start==0:
                        hit=tmp[tmp['plane']==plane][['x','y','mframe','timestamp']]
                    else:
                        hit=np.append(hit,tmp[tmp['plane']==plane][['x','y','mframe','timestamp']])
                    err, buf_out,hit_idx,tlu_idx,event_number=_build(hit,tlu[tlu_idx:],buf,event_number)
                    #print err, len(buf_out),hit_idx,tlu_idx,event_number
                    hit_table.append(buf_out)
                    hit_table.flush()
                    total_data=total_data+len(buf_out)
                    print plane,f_i,"dat=%d"%len(buf_out),"total=%d"%totoal_data,
                    print "%.3f%%"%(100.0*tmpend/end),"%.3fs"%(time.time()-t0)
                    
                    hit=hit[hit_idx:]
                    start=tmpend

if __name__=="__main__":
    import os,sys

    if len(sys.argv)<2:
       print "simple_builder.py <input file> [input file2].."
    fins=sys.argv[1:]

    fout=fins[0][:-3]+"_tlu.h5"
    builde_h5(fins,fout,range(1,7))