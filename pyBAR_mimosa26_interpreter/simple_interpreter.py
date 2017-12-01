''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''
from numba import njit
import numpy as np
import time
import tables

@njit
def _m26_interpreter(raw, dat, idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre,debug): 
    fetot = 0
    fecol = 0
    ferow = 0
    raw_i = 0
    raw_d = 0
    hit = 0
    mid = 0   ## id for mimosa plane 0,1,2,3,4,5
    plane = 0 ## plane number 0:FEI4, 1-6:mimosa, F:TLU
    
    end = len(raw)
    while raw_i < end:
        raw_d = raw[raw_i]
        if (0xFF000000 & raw_d) == 0x20000000: #M26
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x00020000 & raw_d) == 0x20000:
                idx[0] = -1
                idx[1] = -1
                idx[2] = -1
                idx[3] = -1
                idx[4] = -1
                idx[5] = -1
                err=10 ##MIMOSA_DATA_LOSS
                if debug & 0x2 == 0x2:
                   for i in range(6):
                       dat[hit].plane = i+1
                       dat[hit].mframe = mframe[plane]
                       dat[hit].timestamp = timestamp[plane]
                       dat[hit].tlu = tlu
                       dat[hit].x = -2
                       dat[hit].y = err
                       dat[hit].val = 0
                       hit = hit + 1
                #print raw_i,hex(raw_d),mid,idx[mid],"reset frame data because of data loss"
            elif (0x000F0000 & raw_d) == 0x10000:
                if plane==1:
                    ts_pre=timestamp[1]
                    mframe_pre=mframe[1]
                timestamp[plane] = (timestamp[plane] & 0xFFFF0000) | raw_d & 0xFFFF
                idx[mid] = 0
                #print raw_i,hex(raw_d),mid,idx[mid],"frame start"
            elif idx[mid] == -1:
                #print raw_i,hex(raw_d),mid,idx[mid],"trash"
                pass
            else:
                idx[mid] = idx[mid] + 1
                if idx[mid] == 1:
                    if (timestamp[plane]>>16)!=(0x0000FFFF & raw_d) and (((timestamp[plane]>>16)+1) & 0xFFFF)!=(0x0000FFFF & raw_d):
                        err=13 ##timestamp warn 
                        if debug & 0x2 == 0x2:
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = np.uint16(raw_d & 0x0000FFFF)
                            dat[hit].x = 0xFFFE
                            dat[hit].y = err
                            dat[hit].val = 0
                            hit = hit + 1
                    timestamp[plane] = (0x0000FFFF & raw_d) << 16 | timestamp[plane] & 0x0000FFFF
                    #print raw_i,hex(raw_d),mid,idx[mid],"timestamp", timestamp[plane]
                elif idx[mid] == 2:
                    mframe[mid + 1] = (0x0000FFFF & raw_d) | (mframe[plane] & 0xFFFF0000)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 |(mframe[plane] & 0x0000FFFF)
                    if debug & 0x1 == 0x1:
                        dat[hit].plane = plane
                        dat[hit].mframe = mframe[plane]
                        dat[hit].timestamp = timestamp[plane]
                        dat[hit].tlu = tlu
                        dat[hit].x = 0xFFFF
                        dat[hit].y = 0xFFFE
                        dat[hit].val = 0
                        hit = hit + 1
                    #print raw_i,hex(raw_d),mid,idx[mid],"mframe", mframe[plane]
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0x0000FFFF) * 2
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen", dlen[mid]
                elif idx[mid] == 5:
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen2", dlen[mid],(raw_d & 0x0000FFFF) * 2
                    if dlen[mid]!=(raw_d & 0x0000FFFF) * 2:
                        err=3 ##MIMOSA_DLEN_ERROR
                        if debug & 0x2 == 0x2:
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = 0xFFFE
                            dat[hit].y = err
                            dat[hit].val = 0
                            hit = hit + 1
                        return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                elif idx[mid] == 6 + dlen[mid]:
                    #print raw_i,hex(raw_d),mid,idx[mid],"tailer fix value 0xaa50"
                    if raw_d & 0xFFFF != 0xaa50: 
                        err=4 ##MIMOSA_TAILER_ERROR
                        if debug & 0x2 == 0x2:
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = 0xFFFE
                            dat[hit].y = err
                            dat[hit].val = 0
                            hit = hit + 1
                        return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    #print raw_i,hex(raw_d),mid,idx[mid],"tailer2",mframe[plane],plane
                    if raw_d & 0xFFFF != (0xaa50 | plane): 
                        err=5  ##MIMOSA_TAILER2_ERROR
                        if debug & 0x2 == 0x2:
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = 0xFFFE
                            dat[hit].y = err
                            dat[hit].val = 0
                            hit = hit + 1
                        return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                else:
                    if numstatus[mid] == 0:
                        if idx[mid] == 6 + dlen[mid] - 1:
                            pass
                        else:
                            numstatus[mid] = (raw_d) & 0xF
                            row[mid] = (raw_d >> 4) & 0x7FF
                        if raw_d & 0x8000==0x8000:
                            ovf[mid]=ovf[mid]+1
                            numstatus[mid]==0
                            err=8
                            if debug & 0x2 == 0x2:
                                dat[hit].plane = plane
                                dat[hit].mframe = mframe[plane]
                                dat[hit].timestamp = timestamp[plane]
                                dat[hit].tlu = tlu
                                dat[hit].x = 0xFFFE
                                dat[hit].y = err
                                dat[hit].val = 0
                                hit = hit + 1
                                print  "MIMOSA_OVF_WARN",(raw_i,plane)
                            else:
                                return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                        if row[mid]>576:
                            err=1
                            if debug & 0x2 == 0x2:
                                dat[hit].plane = plane
                                dat[hit].mframe = mframe[plane]
                                dat[hit].timestamp = timestamp[plane]
                                dat[hit].tlu = tlu
                                dat[hit].x = 0xFFFE
                                dat[hit].y = err
                                dat[hit].val = 0
                                hit = hit + 1
                            return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col>=1152:
                            err=2
                            if debug & 0x2 == 0x2:
                                dat[hit].plane = plane
                                dat[hit].mframe = mframe[plane]
                                dat[hit].timestamp = timestamp[plane]
                                dat[hit].tlu = tlu
                                dat[hit].x = 0xFFFE
                                dat[hit].y = err
                                dat[hit].val = 0
                                hit = hit + 1
                                print"MIMOSA_COL_ERR",(raw_i,plane)
                            else:
                                return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                        for k in range(num + 1):
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = col + k
                            dat[hit].y = row[mid]
                            dat[hit].val = 0
                            hit = hit + 1
                            
        ###################################
        ###  TLU
        ###################################
        elif (0x80000000 & raw_d) == 0x80000000:
            tlu = raw_d & np.uint16(0xFFFF)
            timestamp[0] = np.uint32((raw_d >>16) & 0x7FFF )| np.uint32(ts_pre & 0xFFFF8000) # TODO be more precise.
            tlu_flg=0
            if (timestamp[0] - ts_pre) & 0x8000 == 0x8000:   ### if timestamp < ts_pre
                timestamp[0]= timestamp[0] + np.uint32(0x8000)
                tlu_flg=1
            #if ((timestamp[0]-ts_pre) & 0x7FFF) > 4608*3:
            #    print "TLU_ERROR maybe a bug in this script",(ts_pre, timestamp[0],(timestamp[0]-ts_pre) & 0x7FFF)
            mframe[0]= mframe_pre + ((timestamp[0]-ts_pre) & 0x7FFF)/4608
            #print mframe_pre,mframe[0], "fix_frame", np.uint32(timestamp[0] - ts_pre)/np.uint32(4608)
            felv1=0
            dat[hit].plane = 0xFF
            dat[hit].mframe = mframe[0]
            dat[hit].timestamp = timestamp[0]
            dat[hit].tlu = tlu
            dat[hit].x = tlu_flg
            dat[hit].y = np.uint16(((timestamp[0]-ts_pre)& 0x7FFF)%4608)
            dat[hit].val = idx[1] ## debug
            hit = hit + 1
            #if np.uint32(timestamp[0] - ts_pre)/np.uint32(4608) > 2:
            #    err=11 #TLU_MFRAME_WARN just print warning
            #    return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                
        ###################################
        ###  FE-I4
        ###################################
        elif (0xFF000000 & raw_d) == 0x01000000: #FEI4
            if( (0xFF0000 & raw_d) == 0x00EA0000) | ((0xFF0000 & raw_d) == 0x00EF0000) |((0xFF0000 & raw_d) == 0x00EC0000): ## other data
                pass
            elif (0xFF0000 & raw_d == 0x00E90000): ##BC
                felv1=felv1+1
                ## TODO get lv1, bc
                bc=raw_d & 0xFF
                lv1=((raw_d & 0x7F00) >> 8)
                lv1 = lv1 ^ (lv1 >> 8);
                lv1 = lv1 ^ (lv1 >> 4);
                lv1 = lv1 ^ (lv1 >> 2);
                lv1 = lv1 ^ (lv1 >> 1);

                if (debug & 0x8) ==0x8:
                    dat[hit].plane = 0
                    dat[hit].mframe = mframe[0]
                    dat[hit].timestamp = timestamp[0]
                    dat[hit].tlu = tlu
                    dat[hit].x = 0xFFFD
                    dat[hit].y = bc
                    dat[hit].val = lv1
                    hit=hit+1
                #if bc-bc0==1 and lv1_0==-1:
                #    bc0=bc
                #    lv1_0=lv1_1
                #elif bc-bc0==0 and lv1_1-lv1_0<16:
                #    pass
                #else:
                #    print "FEI4 header ERROR", hex(raw_d),lv1_1,bc
                #lv=lv1_1-lv_0
                #if debug:
                #    lv1_1=raw_d & 0x7F
                #    bc=((raw_d & 0xFF00)>>8)
                #    print raw_i, hex(raw_d),"FEI4 header","bc=",bc,"lv=",lv1_1 ,lv 
            else: ##TOT1 and TOT2
                fetot=(raw_d & 0x000000F0) >> 4
                fecol=(raw_d & 0x00FE0000) >> 17
                ferow=(raw_d & 0x0001FF00) >> 8
                if fetot !=0xF and fecol<=80 and fecol>=1 and ferow<=336 and ferow>=1:
                    dat[hit].plane = 0
                    dat[hit].mframe = felv1
                    dat[hit].timestamp = timestamp[0]
                    dat[hit].tlu = tlu
                    dat[hit].x = fecol
                    dat[hit].y = ferow
                    dat[hit].val = fetot
                    hit=hit+1
                else:
                    err=6 ## FEI4_TOT1_ERROR
                    if debug & 0x2 == 0x2:
                        dat[hit].plane = plane
                        dat[hit].mframe = mframe[plane]
                        dat[hit].timestamp = timestamp[plane]
                        dat[hit].tlu = tlu
                        dat[hit].x = -2
                        dat[hit].y = err
                        dat[hit].val = 0
                        hit = hit + 1
                    return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                fetot=(raw_d & 0xF)
                ferow=ferow+1
                if fetot!=0xF:
                    if fecol<=80 and fecol>=1 and ferow<=336 and ferow >=1:
                        #dat[hit] = (0,mframe[0],timestamp[0],tlu, fecol, ferow, tot,lv)
                        dat[hit].plane = 0
                        dat[hit].mframe = felv1
                        dat[hit].timestamp = timestamp[0]
                        dat[hit].tlu = tlu
                        dat[hit].x = fecol
                        dat[hit].y = ferow
                        dat[hit].val = fetot
                        hit=hit+1
                    else:
                        err=7 ## FEI4_TOT2_ERROR
                        if debug & 0x2 == 0x2:
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = -2
                            dat[hit].y = err
                            dat[hit].val = 0
                            hit = hit + 1
                        return dat[:hit],raw_i,err,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
        raw_i = raw_i + 1
        
    return dat[:hit],raw_i,0,idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
    
def m26_interpreter(fin,fout,debug=2):
    m26_hit_dtype = np.dtype([('plane', '<u1'),('mframe', '<u4'),('timestamp','<u4'),('tlu', '<u2'),
                      ('x', '<u2'), ('y', '<u2'), ('val','<u1')])
    if isinstance(fin,str):
        fin=[fin]
    
    mframe = np.zeros(7,dtype=np.uint32)
    timestamp = np.zeros(7,dtype=np.uint32)
    dlen = [-1] * 6
    idx = [-1] * 6
    numstatus = [0] * 6
    ovf = [0] * 6
    row = [-1] * 6
    felv1=-1
    tlu = 0
    ts_pre=timestamp[1]
    mframe_pre=mframe[1]
    
    n = 100000000
    dat = np.empty(n, dtype=m26_hit_dtype)
    dat = dat.view(np.recarray)
    t0 = time.time()
    n_fe=0
    n_tlu=0
    with tables.open_file(fout, 'w') as out_file_h5:
        description = np.zeros((1, ), dtype=m26_hit_dtype).dtype
        hit_table = out_file_h5.create_table(out_file_h5.root, 
                        name='Hits', 
                        description=description, 
                        title='hit_data')
        for f_e in fin:
          with tables.open_file(f_e) as tb:
            end=int(len(tb.root.raw_data))
            print "m26_interpreter() fin=%s"%fin
            print "m26_interpreter() # of raw data=%d"%end
            start=0
            while True:
                tmpend=min(start+n,end)
                (hit_dat,raw_i,err,
                    idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre
                    )=_m26_interpreter(
                    tb.root.raw_data[start:tmpend],dat,
                    idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,ts_pre,mframe_pre,
                    debug)
                t1=time.time()-t0
                if err==0:
                    print start,raw_i,len(hit_dat),ovf,"---%.3f%% %.3fs(%.3fus/dat)"%((tmpend*100.0)/end, t1, (t1)/tmpend*1.0E6)
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
                    elif err==11:
                        print "TLU_MFRAME_WARN", np.uint32(timestamp[0] - ts_pre)/np.uint32(4608)
                    print err,start,raw_i,hex(tb.root.raw_data[start+raw_i])
                    if debug & 0x4 == 0x4:
                        print_start=max(start+raw_i-300,0)
                        for j in range(print_start,start+raw_i+100,1):
                            print "ERROR %4d %4d"%(j-start+raw_i,j),hex(tb.root.raw_data[j])
                        break
                    raw_i=raw_i+1
                n_fe=n_fe+len(hit_dat[hit_dat["plane"]==0])
                n_tlu=n_tlu+len(hit_dat[hit_dat["plane"]==255])
                print "n of data: fe=%d"%n_fe,"tlu=%d"%n_tlu
                hit_table.append(hit_dat)
                hit_table.flush()
                start=start+raw_i
                #break
                if start>=end:
                    break
            
if __name__=="__main__":
    import os,sys
    fins=sys.argv[1:]
    for fin in fins:
        fout=fin[:-3]+"_hit.h5"
        m26_interpreter(fin,fout,debug=0x8|0x2|0x1)
        print fout
    #with tables.open_file(fout) as f:
    #    dat=f.root.Hits[:]
    #print dat[dat["plane"]>6]
