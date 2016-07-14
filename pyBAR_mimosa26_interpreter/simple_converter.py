#!/usr/bin/env python
import sys,time,os
from numba import njit
import numpy as np
import tables

hit_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u4'), 
                      ('column', '<u2'), ('row', '<u4'),('frame', '<u4')])
hit_buf_dtype = np.dtype([('frame', '<u4'),('column', '<u2'), ('row', '<u4')])
tlu_buf_dtype = np.dtype([('event_number', '<i8'),('trigger_number', '<u4'), ('frame', '<u4')])

@njit
def _m26_converter(raw_data, plane, hits, mframe, dlen, idx, numstatus, row,ovf,\
               err, tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number,debug):
    hit_i = 0
    jj=0

    for raw_i in range(raw_data.shape[0]):
        raw_d = raw_data[raw_i]
        if (0xFFF00000 & raw_d) == (0x20000000 |(plane <<20)): #M26
            if (0x00020000 & raw_d == 0x20000):
                idx = -1
                #print raw_i,hex(raw_d),mid,idx[mid],"reset frame data because of data loss"
            elif (0x000F0000 & raw_d == 0x10000):
                #timestamp[plane] = raw_d & 0xFFFF
                idx = 0
                #print raw_i,hex(raw_d),mid,idx[mid],"frame start"
            elif idx == -1:
                #print raw_i,hex(raw_d),mid,idx[mid],"trash"
                pass
            else:
                idx = idx + 1
                if idx == 1:
                    pass
                    #timestamp = (0x0000FFFF & raw_d) << 16 |timestamp
                    #print raw_i,hex(raw_d),mid,idx[mid],"timestamp", timestamp[plane]
                elif idx == 2:
                    mframe = (0x0000FFFF & raw_d)
                elif idx == 3:
                    mframe = (0x0000FFFF & raw_d) << 16 | mframe
                    #print raw_i,hex(raw_d),mid,idx[mid],"mframe", mframe[plane]
                elif idx == 4:
                    dlen = (raw_d & 0x0000FFFF) * 2
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen", dlen[mid]
                elif idx == 5:
                    if dlen!=(raw_d & 0x0000FFFF) * 2:
                        return hit_i,raw_i,3 ##MIMOSA_DLEN_ERROR
                elif idx == 6 + dlen:
                    if raw_d & 0xFFFF != 0xaa50: 
                        return hit_i,raw_i,4 ##MIMOSA_TAILER_ERROR
                elif idx == 7 + dlen:  # Last word is frame tailer low word
                    dlen = -1
                    numstatus = 0
                    if raw_d & 0xFFFF != (0xaa50 | plane): 
                        return hit_i,raw_i,5  ##MIMOSA_TAILER2_ERROR
                    ######## copy to hits
                    jj=0
                    for j in range(tlu_buf_i):
                        if tlu_buf[j]["frame"]==mframe-2:
                            for i in range(hit_buf_i):
                                if hit_buf[i]['frame']==mframe-1 or hit_buf[i]['frame']==mframe:
                                    hits[hit_i]["trigger_number"] = tlu_buf[j]["trigger_number"]
                                    hits[hit_i]["event_number"] = tlu_buf[j]["event_number"]
                                    hits[hit_i]['column'] = hit_buf[i]['column']
                                    hits[hit_i]['row'] = hit_buf[i]['row']
                                    hits[hit_i]['frame']=hit_buf[i]['frame']
                                    hit_i=hit_i+1
                                #else :#do nothing        
                        elif tlu_buf[j]['frame']==mframe-1 or tlu_buf[j]['frame']==mframe:
                            tlu_buf[jj]["trigger_number"]=tlu_buf[j]["trigger_number"]
                            tlu_buf[jj]["frame"]=tlu_buf[j]["frame"]
                            tlu_buf[jj]["event_number"]=tlu_buf[j]["event_number"]
                            jj=jj+1
                    tlu_buf_i=jj
                    jj=0
                    for i in range(hit_buf_i):
                        if hit_buf[i]['frame']==mframe:
                            hit_buf[jj]['frame']=hit_buf[i]['frame']
                            hit_buf[jj]['row']=hit_buf[i]['row']
                            hit_buf[jj]['column']=hit_buf[i]['column']
                            hit_buf[jj]['frame']=hit_buf[i]['frame']
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
                            numstatus == 0
                            return hit_i,raw_i,8
                        if row>576:
                            return hit_i,raw_i,1
                    else:
                        numstatus = numstatus - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col>=1152:
                            return hit_i,raw_i,2
                        for k in range(num + 1):
                            hit_buf[hit_buf_i]['frame'] = mframe
                            hit_buf[hit_buf_i]['column'] = col + k
                            hit_buf[hit_buf_i]['row'] = row
                            hit_buf_i = hit_buf_i + 1
        elif(0x80000000 & raw_d == 0x80000000): #TLU
            tlu_buf[tlu_buf_i]["trigger_number"] = raw_d 
            tlu_buf[tlu_buf_i]["frame"] = mframe
            tlu_buf[tlu_buf_i]["event_number"] = event_number
            tlu_buf_i= tlu_buf_i+1
            event_number=event_number+1
    return hit_i,raw_i,0


def m26_converter(fin,fout,plane):
    start=0
    n = 10000000

    mframe = 0
    dlen = -1
    idx = -1
    numstatus = 0
    row = 0
    event_status = 0
    event_number = np.uint64(0)
    hits=np.empty(n*10,dtype=hit_dtype)
    tlu_buf=np.empty(1024,dtype=tlu_buf_dtype)
    hit_buf=np.empty(4096,dtype=hit_buf_dtype)
    err=0
    ovf=0
    tlu_buf_i=0
    hit_buf_i=0
    debug=1

    with tables.open_file(fin) as tb:
        end=int(len(tb.root.raw_data))
        print "fout:",fout,"number of data:",end
        t0 = time.time()
        hit = np.empty(n, dtype=hit_dtype)
        
        with tables.open_file(fout, 'w') as out_file_h5:
            description = np.zeros((1, ), dtype=hit_dtype).dtype
            hit_table = out_file_h5.create_table(out_file_h5.root, 
                        name='Hits', 
                        description=description, 
                        title='hit_data')
            while True:
                tmpend=min(start+n,end)
                tlu_buf_i=0
                hit_i,raw_i,err= _m26_converter(tb.root.raw_data[start:tmpend], plane, hits, mframe, dlen, idx, numstatus,row,ovf,\
                                                   err, tlu_buf,tlu_buf_i,hit_buf,hit_buf_i,event_number,debug)
                t1=time.time()-t0
                if err==0:
                    print start,raw_i,hit_i,err,"---%.3f%% %.3fs(%.3fus/dat)"%((tmpend*100.0)/end, t1, (t1)/tmpend*1.0E6)
                    time.sleep(1)
                else:
                    if err==1:
                        print "MIMOSA_ROW_ERROR",
                    elif err==2:
                        print "MIMOSA_COL_ERROR",
                    elif err==3:
                        print "MIMOSA_DLEN_ERROR",
                    elif err==4:
                        print "MIMOSA_TAILER_ERROR",
                    elif err==5:
                        print "MIMOSA_TAILER2_ERROR",
                    elif err==6:
                        print "FEI4_TOT1_ERROR",
                    elif err==7:
                        print "FEI4_TOT2_ERROR",
                    elif err==8:
                        print "MIMOSA_OVF_WARN",
                    print err,start,raw_i,hex(tb.root.raw_data[start+raw_i])
                    for j in range(-100,100,1):
                        print "ERROR %4d"%j,start+raw_i+j,hex(tb.root.raw_data[start+raw_i+j])
                    break
                hit_table.append(hits[:hit_i])
                hit_table.flush()
                start=start+raw_i+1
                if start>=end:
                    break
if __name__=="__main__":
    import os,sys
    fin=sys.argv[1]
    for i in range(1,7):
        fout=os.path.join(fin[:-3]+"_aligned%d.h5"%i)
        m26_converter(fin,fout,i)

