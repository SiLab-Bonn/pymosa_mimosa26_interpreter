#!/usr/bin/env python
import sys,time,os
from numba import njit
import numpy as np
import tables

hit_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u2'),('event_timestamp','<u2'),('frame_timestamp','<u4'), 
                      ('frame','<u4'),('x', '<u2'), ('y', '<u2')])
hit_buf_dtype = np.dtype([('frame', '<u4'),('x', '<u2'), ('y', '<u2'),('timestamp','<u4')])
tlu_buf_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u2'), ('frame', '<u4'),('timestamp','<u2')])

@njit
def _m26_builder(raw_data, plane, hits, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number,debug):
    hit_i = 0
    jj=0

    for raw_i in range(raw_data.shape[0]):
        raw_d = raw_data[raw_i]
        if (0xFFF00000 & raw_d) == (0x20000000 |(plane <<20)): #M26
            #print raw_i,hex(raw_d),idx
            if (0x00020000 & raw_d == 0x20000):
                idx = -1
                #print raw_i,hex(raw_d),mid,idx[mid],"reset frame data because of data loss"
            elif (0x000F0000 & raw_d == 0x10000):
                timestamp = raw_d & 0xFFFF | timestamp & 0xFFFF0000
                idx = 0
                #print raw_i,hex(raw_d),"frame start"
            elif idx == -1:
                #print raw_i,hex(raw_d),mid,idx[mid],"trash"
                pass
            else:
                idx = idx + 1
                if idx == 1:
                    timestamp = (0x0000FFFF & raw_d) << 16 | (timestamp & 0xFFFF)
                    #print raw_i,hex(raw_d),mid,idx[mid],"timestamp", timestamp[plane]
                elif idx == 2:
                    mframe = (0x0000FFFF & raw_d)| (0xFFFF0000 & mframe)
                elif idx == 3:
                    mframe = (0x0000FFFF & raw_d) << 16 | (mframe & 0xFFFF)
                    #print raw_i,hex(raw_d),idx,"mframe", mframe
                elif idx == 4:
                    dlen = (raw_d & 0x0000FFFF) * 2
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen", dlen[mid]
                elif idx == 5:
                    if dlen!=(raw_d & 0x0000FFFF) * 2:
                        return hits[:hit_i],raw_i,3, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                               tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
                elif idx == 6 + dlen:
                    if raw_d & 0xFFFF != 0xaa50: 
                        return hits[:hit_i],raw_i,4, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                               tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
                elif idx == 7 + dlen:  # Last word is frame tailer low word
                    dlen = -1
                    numstatus = 0
                    if raw_d & 0xFFFF != (0xaa50 | plane): 
                        return hits[:hit_i],raw_i,5,mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                               tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
                    ######## copy to hits
                    jj=0
                    for j in range(tlu_buf_i):
                        if tlu_buf[j]["frame"]==mframe-2:
                            for i in range(hit_buf_i):
                                if hit_buf[i]['frame']==mframe-1 or hit_buf[i]['frame']==mframe:
                                    hits[hit_i]["trigger_number"] = tlu_buf[j]["trigger_number"]
                                    hits[hit_i]["event_number"] = tlu_buf[j]["event_number"]
                                    hits[hit_i]["event_timestamp"] = tlu_buf[j]["timestamp"]
                                    hits[hit_i]['x'] = hit_buf[i]['x']
                                    hits[hit_i]['y'] = hit_buf[i]['y']
                                    hits[hit_i]['frame']=hit_buf[i]['frame']
                                    hits[hit_i]['frame_timestamp']=hit_buf[i]['timestamp']
                                    hit_i=hit_i+1
                                #else :#do nothing        
                        elif tlu_buf[j]['frame']==mframe-1 or tlu_buf[j]['frame']==mframe:
                            tlu_buf[jj]["trigger_number"]=tlu_buf[j]["trigger_number"]
                            tlu_buf[jj]["frame"]=tlu_buf[j]["frame"]
                            tlu_buf[jj]["event_number"]=tlu_buf[j]["event_number"]
                            tlu_buf[jj]["timestamp"]=tlu_buf[j]["timestamp"]
                            jj=jj+1
                    tlu_buf_i=jj
                    jj=0
                    for i in range(hit_buf_i):
                        if hit_buf[i]['frame']==mframe:
                            hit_buf[jj]['frame']=hit_buf[i]['frame']
                            hit_buf[jj]['y']=hit_buf[i]['y']
                            hit_buf[jj]['x']=hit_buf[i]['x']
                            hit_buf[jj]['timestamp']=hit_buf[i]['timestamp']
                            jj=jj+1
                    hit_buf_i=jj
                    if hit_i > hits.shape[0]-1000:
                        break
                else:
                    if numstatus == 0:
                        if idx == 6 + dlen - 1:
                            pass
                        else:
                            numstatus = (raw_d) & 0xF
                            row = (raw_d >> 4) & 0x7FF
                        if raw_d & 0x8000==0x8000:
                            ovf=ovf+1
                            numstatus = 0
                            return hits[:hit_i],raw_i,8, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number

                        if row>576:
                            return hits[:hit_i],raw_i,1, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
                    else:
                        numstatus = numstatus - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col==0x5C0:
                            return hits[:hit_i],raw_i,9, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
 ##MIMOSA_COL_OVF?_WARN
                        elif col>=1152:
                            return hits[:hit_i],raw_i,2, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
 ##MIMOSA_COL_ERROR

                        for k in range(num + 1):
                            if col+k >=1152:
                                return hits[:hit_i],raw_i,12, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                                   tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
                            hit_buf[hit_buf_i]['frame'] = mframe
                            hit_buf[hit_buf_i]['timestamp'] = timestamp
                            hit_buf[hit_buf_i]['x'] = col + k
                            hit_buf[hit_buf_i]['y'] = row
                            hit_buf_i = hit_buf_i + 1
        elif(0x80000000 & raw_d == 0x80000000): #TLU
            tlu_buf[tlu_buf_i]["trigger_number"] = raw_d & 0xFFFF 
            tlu_buf[tlu_buf_i]["timestamp"] = (raw_d>>16) & 0x7FFF 
            tlu_buf[tlu_buf_i]["frame"] = mframe
            tlu_buf[tlu_buf_i]["event_number"] = event_number
            #rint raw_i,hex(raw_d),"tlu",mframe, raw_d & 0x0000FFFF ,event_number
            tlu_buf_i= tlu_buf_i+1
            if tlu_buf_i==tlu_buf.shape[0]:
                return hits[:hit_i],raw_i,10, mframe,timestamp,dlen, idx, numstatus, row,ovf,\
                       tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number
  ##TLU_BUF_OVF_WANNING
            event_number=event_number+1

    return hits[:hit_i],raw_i,0,mframe,timestamp,dlen, idx, numstatus, row,ovf,\
           tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number

def m26_builder_h5(fin,fout,plane):
    n = 100000000
    if isinstance(fin,str):
      fin=[fin]

    mframe = 0
    timestamp = 0
    dlen = -1
    idx = -1
    numstatus = 0
    row = 0
    event_status = 0
    event_number = np.uint64(0)
    buf = np.empty(n,dtype=hit_dtype)
    tlu_buf = np.empty(1024,dtype=tlu_buf_dtype)
    hit_buf = np.empty(4096,dtype=hit_buf_dtype)
    ovf = 0
    tlu_buf_i = 0
    hit_buf_i = 0
    debug = 1

    t0 = time.time()
    with tables.open_file(fout, 'w') as out_file_h5:
        description = np.zeros((1, ), dtype=hit_dtype).dtype
        hit_table = out_file_h5.create_table(out_file_h5.root, 
                    name='Hits', 
                    description=description, 
                    title='hit_data')
        for fin_e in fin:
          with tables.open_file(fin_e) as tb:
            end=int(len(tb.root.raw_data))
            print "fin:",fin_e,"number of data:",end
            start=0
            while True:
                tmpend=min(start+n,end)
                hits,raw_i,err, mframe,timestamp, dlen, idx, numstatus, row,ovf, \
                      tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number = \
                      _m26_builder(tb.root.raw_data[start:tmpend], plane, buf, mframe,timestamp,dlen, idx, numstatus,row,ovf,\
                      tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number,debug)
                t1=time.time()-t0
                if err==0:
                    print start,raw_i,err,"---%.3f%% %.3fs(%.3fus/dat)"%((tmpend*100.0)/end, t1, (t1)/tmpend*1.0E6)
                else: ## Fix error code
                    if err==1:
                        err_str= "MIMOSA_ROW_ERROR"
                    elif err==2:
                        err_str= "MIMOSA_COL_ERROR"
                    elif err==3:
                        err_str= "MIMOSA_DLEN_ERROR"
                    elif err==4:
                        err_str= "MIMOSA_TAILER_ERROR"
                    elif err==5:
                        err_str= "MIMOSA_TAILER2_ERROR"
                    elif err==6:
                        err_str= "FEI4_TOT1_ERROR"
                    elif err==7:
                        err_str= "FEI4_TOT2_ERROR"
                    elif err==8:
                        err_str= "MIMOSA_OVF_WARN"
                    elif err==9:
                        err_str= "MIMOSA_COL_OVF?_WARN"
                    elif err==10:
                        err_str= "TLU_BUF_OVF_WANNING"
                        tlu_buf_i=125
                        tlu_buf[:125]=tlu_buf[-125:]
                    elif err==12:
                        err_str= "MIMOSA_COL_ERROR2"
                    print err_str,start,raw_i,hex(tb.root.raw_data[start+raw_i])
                    #for j in range(-100,100,1):
                    #    print "ERROR %4d"%j,start+raw_i+j,hex(tb.root.raw_data[start+raw_i+j])
                    #break
                hit_table.append(hits)
                hit_table.flush()
                start=start+raw_i+1
                if start>=end:
                    break

if __name__=="__main__":
    import os,sys

    if len(sys.argv)<2:
       print "simple_builder.py <input file> [input file2].."
    fin=sys.argv[1:]

    for i in range(1,7):
        print "------------",i,"---------------"
        fout=fin[0][:-3]+"_tlu%d.h5"%i
        m26_builder_h5(fin,fout,i)
