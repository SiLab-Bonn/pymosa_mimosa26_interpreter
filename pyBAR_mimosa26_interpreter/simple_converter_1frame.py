import sys, time, os
from numba import njit

import numpy as np
import tables

hit_type= [("event_number","<i8"),("frame","<u1"),("column","<u2"),("row","<u2"),("charge","<u2")]
ret_type = [('event_number', '<i8'), ('event_timestamp', '<u4'),('trigger_number','<u4'),('mframe',"<u4"),('m_timestamp','<u4'),
            ("x",'<u2'),("y",'<u4'),("tlu_y",'<u4'),('tlu_mframe',"<u4")]

@njit
def _correlate_range(dat1_begin, dat1_end, dat2, buf, mode,debug=0):
    """ order by begin """
    d1_idx=0
    d2_idx=0
    buf_i=0
    while d1_idx < len(dat1_begin) and d2_idx < len(dat2):
        if dat1_begin[d1_idx]> dat2[d2_idx]:
            if (debug & 0x8)==0x8:
                print d1_idx,d2_idx,dat1_begin[d1_idx], dat1_end[d1_idx],dat2[d2_idx],
                print dat1_begin[d1_idx]/4608, dat1_end[d1_idx]/4608,dat2[d2_idx]/4608,
                print "next2"
            #break
            d2_idx=d2_idx+1
        elif dat1_end[d1_idx] < dat2[d2_idx]:
            if (debug & 0x8)==0x8:
                print d1_idx,d2_idx,dat1_begin[d1_idx], dat1_end[d1_idx],dat2[d2_idx],
                print dat1_begin[d1_idx]/4608, dat1_end[d1_idx]/4608,dat2[d2_idx]/4608,
                print "next1"
            d1_idx=d1_idx+1
        else:
            if (debug & 0x8)==0x8:
                print d1_idx,d2_idx,dat1_begin[d1_idx],dat1_end[d1_idx],dat2[d2_idx],
                print "found"
            for c1_i,d1_begin in enumerate(dat1_begin[d1_idx:]):
                if d1_begin >  dat2[d2_idx]:
                    break
            if c1_i+1 == len(dat1_begin[d1_idx:]) and (mode & 0x4) == 0x0:
                #print "read data1 more"
                return 1,buf[:buf_i],d1_idx,d2_idx
            if len(buf)-buf_i < c1_i:
                #print "buffer full"
                return 3,buf[:buf_i],d1_idx,d2_idx

            for d1_i in range(c1_i):
                if dat2[d2_idx] < dat1_end[d1_idx+d1_i]:
                    buf[buf_i]["idx1"]=d1_idx+d1_i
                    buf[buf_i]["idx2"]=d2_idx
                    buf[buf_i]["id1_begin"]=dat1_begin[d1_idx+d1_i]
                    buf[buf_i]["id1_end"]=dat1_end[d1_idx+d1_i]
                    buf[buf_i]["id2"]=dat2[d2_idx]
                    buf_i=buf_i+1                
            d2_idx=d2_idx+1

    if d2_idx < len(dat2):
         # print read data2 more
         return 2,buf[:buf_i],d1_idx,d2_idx
    elif d1_idx < len(dat1_begin):
         # print read data2 more
         return 1,buf[:buf_i],d1_idx,d2_idx
    #return 0,buf[:buf_i],d1_idx,d2_idx
    
@njit
def _correlate(dat1,dat2,buf,mode,mask):
    d1_idx=0
    d2_idx=0
    buf_i=0
    #print mode & 0x3
    while d1_idx < len(dat1) and d2_idx < len(dat2):
        #print d1_idx,d2_idx,dat1[d1_idx],dat2[d2_idx],
        if ((dat1[d1_idx]-dat2[d2_idx]) & mask)==mask:
            #print "next1"
            d1_idx=d1_idx+1
        elif ((dat2[d2_idx]-dat1[d1_idx]) & mask)==mask :
            #print "next2"
            d2_idx=d2_idx+1
        else: ##dat1[d1_idx]==dat2[d2_idx]
            for d1_i,d1 in enumerate(dat1[d1_idx:]):
                if dat1[d1_idx]!=d1:
                    break
            for d2_i,d2 in enumerate(dat2[d2_idx:]):
                if dat2[d2_idx]!=d2:
                    break
            #print "eq",d1_i,d2_i
            
            if dat1[d1_idx]==d1:
                if (mode & 0x4) ==0x4:
                    d1_i=d1_i+1
                else:
                    #print "more dat1",(dat1[d1_idx],dat1[-1],dat2[d2_idx],dat2[-1])
                    return 1,buf[:buf_i],d1_idx,d2_idx
            if dat2[d2_idx]==d2:
                if (mode & 0x4) == 0x4:
                    d2_i=d2_i+1
                else:
                    #print "more dat2",(dat1[d1_idx],dat1[-1],dat2[d2_idx],dat2[-1])
                    return 2,buf[:buf_i],d1_idx,d2_idx
            if len(buf)-buf_i <= d1_i*d2_i:
                #print "buffer full",(len(buf),buf_i,d1_i,d2_i)
                return 3,buf[:buf_i],d1_idx,d2_idx

            if (mode & 0x3) ==0x1:
                #print "mode1",d1_i,d2_i
                for c1_i in range(d1_i):
                    for c2_i in range(d2_i):
                        #print idx1[u1_i],idx2[u2_i],c1_i,c2_i,"event_number",hit1[idx1[u1_i]+c1_i],hit2[idx2[u2_i]+c2_i]
                        buf[buf_i]["idx1"]=d1_idx+c1_i
                        buf[buf_i]["idx2"]=d2_idx+c2_i
                        buf[buf_i]["id"]=dat2[d2_idx]
                        buf_i=buf_i+1                
            elif (mode & 0x3) ==0x2:
                for c_i in range(max(d1_i,d2_i)):
                    if c_i<d1_i:
                        buf[buf_i]["idx1"]=d1_idx+c_i
                    else:
                        buf[buf_i]["idx1"]=len(dat1)
                    if c_i<d2_i:
                        buf[buf_i]["idx2"]=d2_idx+c_i
                    else:
                        buf[buf_i]["idx2"]=len(dat2)
                    buf[buf_i]["id"]=dat2[d2_idx]
                    buf_i=buf_i+1
            elif (mode & 0x3) == 0x3:
                #print "mode3",d1_idx,d2_idx
                for c1_i in range(d1_i):
                    buf[buf_i]["idx1"]=d1_idx+c1_i
                    buf[buf_i]["idx2"]=d2_idx
                    buf[buf_i]["id"]=dat2[d2_idx]
                    buf_i=buf_i+1
            d1_idx=d1_idx+d1_i
            d2_idx=d2_idx+d2_i
    if d2_idx < len(dat2):
         # print read data2 more
         return 2,buf[:buf_i],d1_idx,d2_idx
    elif d1_idx < len(dat1):
         # print read data2 more
         return 1,buf[:buf_i],d1_idx,d2_idx
    return 0,buf[:buf_i],d1_idx,d2_idx

def build_h5(fin,fout,plane=1,lowerlim=-48,upperlim=0,n=10000000,debug=0):
    if isinstance(fin,str):
        fin=[fin]
    event_number=np.int64(0)
    tlu=np.empty(0,dtype=[('mframe', '<u4'),('timestamp','<u4'),('tlu', '<u2'), ('y', '<u2')])
    m26=np.empty(0,dtype=[('mframe', '<u4'),('timestamp','<u4'),('x', '<u2'), ('y', '<u2')])
    idx_type=[("id1_begin","<i8"),("id1_end","<i8"),("id2","<i8"),("idx1","<u8"),("idx2","<u8")]
    buf=np.empty(n,dtype=idx_type)
    
    with tables.open_file(fout,"w") as f_o:
        description = np.zeros((1, ), dtype=ret_type).dtype
        hit_table = f_o.create_table(f_o.root, name='Hits',description=description,title='hit_data')
        t0=time.time()
        total=0
        for fin_i,fin_e in enumerate(fin):
            with tables.open_file(fin_e) as f:
                end=len(f.root.Hits)
                start=0
                while start<end:
                    tmpend=min(end,start+n)
                    hit=f.root.Hits[start:tmpend]
                    tlu=np.append(tlu,hit[hit["plane"]==0xFF][["mframe","timestamp","tlu","y"]])
                    m26=np.append(m26,hit[np.bitwise_and(hit["plane"]==plane,hit["x"]<1152)][["mframe","timestamp","x","y"]])

                    if (debug & 0x8) ==0x8:
                    #if True:
                        print "m26",m26["mframe"]
                        print m26[m26['x']>1151]
                        print "tlu",tlu["mframe"]

                    #################
                    ## check data
                    arg=np.argwhere(tlu["mframe"][1:]<tlu["mframe"][:-1])
                    if len(arg)==0 and (debug & 0x8)==0x8:
                        print "build_h5() tlu mframe always increase True"
                    elif start==0 and fin_i==0 and len(arg)==1:
                        print "build_h5() tlu mframe decrease once, cut before idx=%d"%arg[0][0]
                        tlu=tlu[arg[0][0]+1:]
                        event_number=arg[0][0]+1
                    else:
                        for a_i, a in enumerate(arg):
                           print a_i,"build_h5() ERROR!!tlu-mframe decrease at idx=%d fix data"%a[0],tlu[a[0]],tlu[a[0]+1]
                           if a_i==10:
                              print "more... %d in total"%len(arg)

                    #################
                    ## correlate data
                    print "build_h5() %d corrleate #_tlu=%d, #_m26=%d"%(start,len(tlu),len(m26)),
                    
                    dat1_begin=np.int64(m26["mframe"])*4608+np.int64(m26["y"])*8-2*4608+lowerlim
                    dat1_end=dat1_begin-lowerlim+4608+upperlim
                    dat2=np.int64(tlu['mframe'])*4608+np.int64(tlu["y"])
                    if fin_e==fin[-1] and tmpend==end:
                       mode=4
                    else:
                       mode=0
                    err,buf_out,m26_idx,tlu_idx=_correlate_range(dat1_begin, dat1_end, dat2, buf, mode,debug=debug)
                    ret=np.empty(len(buf_out),dtype=ret_type)
                    ret['event_number']=buf_out["idx2"]+event_number
                    ret['event_timestamp']=tlu[buf_out["idx2"]]["timestamp"]
                    ret['trigger_number']=tlu[buf_out["idx2"]]["tlu"]
                    ret['tlu_mframe']=tlu[buf_out["idx2"]]["mframe"]
                    ret['tlu_y']=tlu[buf_out["idx2"]]["y"]
                    ret['mframe']=m26[buf_out["idx1"]]["mframe"]
                    ret['m_timestamp']=m26[buf_out["idx1"]]["timestamp"]
                    ret['x']=m26[buf_out["idx1"]]["x"]
                    ret['y']=m26[buf_out["idx1"]]["y"]

                    hit_table.append(ret)
                    hit_table.flush()
                    total=total+len(ret)
                    #print "============",ret["x"][ret["x"]>1151]
                    print "colleated=%d status=%d"%(len(ret),err),"%.3f%%"%(100.0*tmpend/end),"%.3fs"%(time.time()-t0)

                    event_number=event_number+tlu_idx
                    tlu=tlu[tlu_idx:]
                    m26=m26[m26_idx:]
                    start=tmpend
                    if debug & 0x1==0x1 and total> 1500000:
                        break
        hit_table.attrs.upperlim=upperlim
        hit_table.attrs.lowerlim=lowerlim
        f_o.close()
    print "build_h5() DONE fout= %s"%fout,"%.3fs"%(time.time()-t0)
    
def convert_h5(fin,fref,fout,x_offset=1,x_factor=1,y_offset=1,y_factor=1,tr=False,n=1000000,debug=0):
    with tables.open_file(fref) as f:
        ref=f.root.Hits[:][["event_number","trigger_number"]]
    print "convert_h5() fref= %s"%fref
    print "convert_h5() # of ref data=%d"%len(ref)
    buf_type=[("id","<u8"),("idx1","<u8"),("idx2","<u8")]
    buf_idx=np.empty(n+len(ref),dtype=buf_type)
    with tables.open_file(fout,"w") as f_o:
        t0=time.time()
        description = np.zeros((1, ), dtype=hit_type).dtype
        hit_table = f_o.create_table(f_o.root, name='Hits',description=description,title='hit_data')
        
        with tables.open_file(fin) as f:
            end=len(f.root.Hits)
            start=0
            print "convert_h5() fin= %s"%fin
            print "convert_h5() #_data=%d"%end
            while start<end:
                #tmpend=min(end,start+n)
                tmpend=end
                hit=f.root.Hits[start:tmpend][["x","y",'trigger_number']]
                #print "============",hit[hit['x']>1151]
                err,buf_idx,hit_idx,ref_idx=_correlate(hit["trigger_number"],ref["trigger_number"],buf_idx,mode=0x7,mask=0x4000)
                hit=hit[buf_idx["idx1"]]
                #print "============",hit[hit['x']>1151]
                buf=np.empty(len(hit),dtype=hit_type)      
                if tr==True:
                    buf[:len(hit)]['row']=x_offset+x_factor*hit['x']
                    buf[:len(hit)]['column']=y_offset+y_factor*hit['y']
                else:
                    buf[:len(hit)]['column']=x_offset+x_factor*hit['x']
                    buf[:len(hit)]['row']=y_offset+y_factor*hit['y']

                buf[:len(hit)]['frame']=np.zeros(len(hit),dtype=np.uint8)
                buf[:len(hit)]['charge']=np.ones(len(hit),dtype=np.uint16)
                buf[:len(hit)]['event_number']=ref[buf_idx["idx2"]]['event_number']

                #print x_offset, x_factor
                #print "============", buf[:len(hit)][buf[:len(hit)]['column']>1152]
                hit_table.append(buf[:len(hit)])
                hit_table.flush()
                
                print "convert_h5() %.3f%%done"%(100.*tmpend/end),"%.3fs"%(time.time()-t0)
                
                start=tmpend
       
if __name__=="__main__":
  import os,sys
  fins=sys.argv[1:]
  fins="/sirrush/thirono/testbeam/2017-11-08/run166/166_elsa_20171108_m26_telescope_scan_hit.h5"
  for i in range(1,7):
      i=5
      print "----------",i,"----------"
      #for ii in range(-20,20,2):
      #    fout=fins[0][:-7]+"_tlu%d-%d.h5"%(i,ii)
      fout=fins[:-7]+"_tlu%d.h5"%i
      #build_h5(fins,fout,plane=1,lowerlim=-6*8,upperlim=0*8,debug=0)
      
      fin=fout
      fref=fins[:-7]+"_tlu.h5"
      fout=fins[:-7]+"_ev%d.h5"%i
      convert_h5(fin,fref,fout,x_offset=1,x_factor=1,y_offset=1,y_factor=1,tr=False,n=1000000,debug=0)
      break
