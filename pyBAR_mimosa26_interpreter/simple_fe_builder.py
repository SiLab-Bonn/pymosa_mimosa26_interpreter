''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''

from numba import njit
import numpy as np
import time
import tables

fe_dtype=[('event_number', '<i8'), ('trigger_number', '<u4'), ('trigger_time_stamp', '<u8'), 
          ('relative_BCID', '<u1'), ('BCID', '<u1'),('LV1ID', '<u1'),('column', '<u1'), ('row', '<u2'), ('tot', 'u1')]
hit_dtype=[('plane', 'u1'), ('mframe', '<u4'), ('timestamp', '<u4'), ('tlu', '<u2'), ('x', '<u2'), ('y', '<u2'), ('val', 'u1')]

@njit
def _build_fe(hit,buf,h_idx,flg,bc0,bc,lv1,event_number,tlu,timestamp,debug):
    h_i=0
    buf_i=0
    lv1_debug=0
    while h_i< len(hit):
        if flg==1 or flg==0:
            ii=0
            for ii,hh in enumerate(hit[h_idx:]):
                if hh["plane"]==255:
                    pass
                elif hh["x"]==0xFFFD:
                    
                    if flg==0:
                        lv1=hh["val"]
                        bc0=hh["y"]
                        flg=1
                    elif lv1!=hh["val"]:
                        flg=2
                        break
                    bc=hh["y"]
                    lv1_debug=hh["y"]
                elif flg==1:
                    buf[buf_i]["event_number"]=max(event_number,0)
                    buf[buf_i]["trigger_number"]=tlu
                    buf[buf_i]["trigger_time_stamp"]=np.uint64(timestamp)
                    buf[buf_i]["relative_BCID"]= np.uint8(bc-bc0)
                    buf[buf_i]["BCID"]= bc
                    buf[buf_i]["LV1ID"]= lv1
                    buf[buf_i]["column"]=hh['x']
                    buf[buf_i]["row"]=hh['y']
                    buf[buf_i]["tot"]=hh['val']
                    buf_i=buf_i+1         
            if flg==2:
                if ii > 1000:
                    print "WARN too big event",(h_i,h_idx,ii)
                h_idx=h_idx+ii
            else:
                h_idx=h_idx+ii+1
                return buf[:buf_i],h_i,h_idx,flg,bc0,bc,lv1,event_number,tlu,timestamp
        elif flg==2:
            if hit[h_i]["plane"]==255:
                    h_idx=max(h_i+1,h_idx)
                    flg=0
                    tlu=hit[h_i]['tlu']
                    timestamp=hit[h_i]['timestamp']
                    event_number=event_number+1
                    if debug==1:
                        buf[buf_i]["event_number"]=event_number
                        buf[buf_i]["trigger_number"]=tlu
                        buf[buf_i]["trigger_time_stamp"]=np.uint64(timestamp)
                        buf[buf_i]["relative_BCID"]= 0
                        buf[buf_i]["BCID"]= np.uint8(h_idx-h_i +1)
                        buf[buf_i]["LV1ID"]=0
                        buf[buf_i]["column"]=0
                        buf[buf_i]["row"]=0
                        buf[buf_i]["tot"]=0
                        buf_i=buf_i+1  
            else:
                pass
            h_i=h_i+1

             
    return buf[:buf_i],h_i,h_idx,flg,bc0,bc,lv1,event_number,tlu,timestamp
            
def build_fe_h5(fins,fout,n=100000000,debug=1):
    """
    debug: 0x1= tlu data in (col,row)=(0,0)
    """
    if isinstance(fins,str):
         fins=[fins]
    hit=np.empty(0,dtype=hit_dtype)
    buf=np.empty(n,dtype=fe_dtype)
    event_number=np.int64(-1)
    lv1=np.uint8(0xFF)
    bc0=np.uint8(0xFF)
    bc=np.uint8(0)
    flg=2
    h_idx=0
    tlu=np.uint32(0)
    timestamp=np.uint32(0)
    pre_timestamp=np.uint64(0)
    with tables.open_file(fout, "w") as f_o:
        description=np.zeros((1,),dtype=fe_dtype).dtype
        hit_table=f_o.create_table(f_o.root,name="Hits",description=description,title='hit_data')
        t0=time.time()
        total_data=0
        for fin in fins:
            with tables.open_file(fin) as f:
                start=0
                end=len(f.root.Hits)
                print "build_fe2_h5() fin",fin,"# of data",end
                while start < end:
                    tmpend=min(end,start+n)
                    tmp=f.root.Hits[start:tmpend]
                    hit=np.append(hit,tmp[np.bitwise_or(tmp['plane']==255,tmp['plane']==0)])
                    if len(hit)==0:
                        start=tmpend
                        continue

                    (buf_out,h_i,h_idx,flg,bc0,bc,lv1,event_number,tlu,timestamp
                        )=_build_fe2(
                         hit,buf,h_idx,flg,bc0,bc,lv1,event_number,tlu,timestamp,debug=debug)

                        hit_table.append(buf_out)
                        hit_table.flush()
                        pre_timestamp=buf_out["trigger_time_stamp"][-1]
                    hit=hit[h_i:]
                    h_idx=h_idx-h_i
                    total_data=total_data+len(buf_out)
                    print "%.3f%%"%(100.0*tmpend/end),"%.3fs"%(time.time()-t0), "total_data=%d"%total_data
                    start=tmpend
    
if __name__=="__main__":
    import os,sys,string
    if len(sys.argv)<2:
        print "simple_fe_builder.py <hit_file> [hit_file]..."
        print "output is *tlu.h5"
    fins=[]
    for fin in sys.argv[1:]:
          fins.append(fin)
    fout=string.join(fins[0].split("_")[:-1],"_")+"_tlu.h5"
    build_fe_h5(fins,fout,n=100000000)

