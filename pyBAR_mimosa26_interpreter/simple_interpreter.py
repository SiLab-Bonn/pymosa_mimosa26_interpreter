''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''

from numba import njit
import numpy as np
import time
import tables

@njit
def _m26_interpreter(raw, dat, idx,mframe,timestamp,dlen,numstatus,row,tlu,lv1,ovf,tlu_flg): 
    fetot = 0
    fecol = 0
    ferow = 0
    felv1=0
    raw_i = 0
    raw_d = 0
    hit = 0
    mid = 0   ## id for mimosa plane 0,1,2,3,4,5
    plane = 0 ## plane number 0:FEI3, 1-6:mimosa, F:TLU
    
    end = len(raw)
    
    while raw_i < end:
        raw_d = raw[raw_i]
        if (0xFF000000 & raw_d == 0x20000000): #M26
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x00020000 & raw_d == 0x20000):
                idx[0] = -1
                idx[1] = -1
                idx[2] = -1
                idx[3] = -1
                idx[4] = -1
                idx[5] = -1
                #print raw_i,hex(raw_d),mid,idx[mid],"reset frame data because of data loss"
            elif (0x000F0000 & raw_d == 0x10000):
                timestamp[plane] = raw_d & 0xFFFF
                idx[mid] = 0
                #print raw_i,hex(raw_d),mid,idx[mid],"frame start"
            elif idx[mid] == -1:
                #print raw_i,hex(raw_d),mid,idx[mid],"trash"
                pass
            else:
                idx[mid] = idx[mid] + 1
                if idx[mid] == 1:
                    timestamp[plane] = (0x0000FFFF & raw_d) << 16 | timestamp[plane]
                    #print raw_i,hex(raw_d),mid,idx[mid],"timestamp", timestamp[plane]
                elif idx[mid] == 2:
                    mframe[mid + 1] = (0x0000FFFF & raw_d)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 | mframe[plane]
                    #print raw_i,hex(raw_d),mid,idx[mid],"mframe", mframe[plane]
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0x0000FFFF) * 2
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen", dlen[mid]
                elif idx[mid] == 5:
                    #print raw_i,hex(raw_d),mid,idx[mid],"dlen2", dlen[mid],(raw_d & 0x0000FFFF) * 2
                    if dlen[mid]!=(raw_d & 0x0000FFFF) * 2:
                        return dat[:hit],raw_i,3 ##MIMOSA_DLEN_ERROR
                elif idx[mid] == 6 + dlen[mid]:
                    #print raw_i,hex(raw_d),mid,idx[mid],"tailer fix value 0xaa50"
                    if raw_d & 0xFFFF != 0xaa50: 
                        return dat[:hit],raw_i,4 ##MIMOSA_TAILER_ERROR
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    #print raw_i,hex(raw_d),mid,idx[mid],"tailer2",mframe[plane],plane
                    if raw_d & 0xFFFF != (0xaa50 | plane): 
                        return dat[:hit],raw_i,5  ##MIMOSA_TAILER2_ERROR
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
                            return dat[:hit],raw_i,8
                        if row[mid]>576:
                            return dat[:hit],raw_i,1
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col>=1152:
                            return dat[:hit],raw_i,2
                        for k in range(num + 1):
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = tlu
                            dat[hit].x = col + k
                            dat[hit].y = row[mid]
                            dat[hit].val = 0
                            dat[hit].val2 = 0
                            hit = hit + 1
        elif(0x80000000 & raw_d == 0x80000000): #TLU
            tlu = raw_d & 0xFFFF
            timestamp[0] = (raw_d >>16) & 0x7FFF | (timestamp[1] & 0xFFFF8000) # TODO be more precise.
            if timestamp[0] < timestamp[1]:
                  timestamp[0]= timestamp[0] + 0x8000
                  tlu_flg=1
            mframe[0] = mframe[1]
            felv1=0
            dat[hit].plane = -1
            dat[hit].mframe = mframe[0]
            dat[hit].timestamp = timestamp[0]
            dat[hit].tlu = tlu
            dat[hit].x = 0
            dat[hit].y = 0
            dat[hit].val = idx[1] ## debug
            dat[hit].val2 = tlu_flg
            hit = hit + 1
            tlu_flg=0
     
        elif(0xFF000000 & raw_d == 0x01000000): #FEI4
            if(0xFF0000 & raw_d == 0x00EA0000) | (0xFF0000 & raw_d == 0x00EF0000) |(0xFF0000 & raw_d == 0x00EC0000): ## other data
                pass
            elif (0xFF0000 & raw_d == 0x00E90000): ##BC
                felv1=felv1+1
                ## TODO get lv1, bc
                #bc=((raw_d & 0xFF00)>>8)
                #lv1_1=raw_d & 0x7F
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
                    dat[hit].mframe = mframe[0]
                    dat[hit].timestamp = timestamp[0]
                    dat[hit].tlu = tlu
                    dat[hit].x = fecol
                    dat[hit].y = ferow
                    dat[hit].val = fetot
                    dat[hit].val2 = felv1
                    hit=hit+1
                else:
                    #pass
                    return dat[:hit],raw_i,6 ## FEI4_TOT1_ERROR
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
                        dat[hit].val = felv1
                        hit=hit+1
                    else:
                        return dat[:hit],raw_i,7 ##FEI4_TOT2_ERROR
        raw_i = raw_i + 1
        
    return dat[:hit],raw_i,0
    
def m26_interpreter(fin,fout,debug=0):
    m26_hit_dtype = np.dtype([('plane', '<u1'),('mframe', '<u4'),('timestamp','<u4'),('tlu', '<u1'),
                      ('x', '<u2'), ('y', '<u2'), ('val','<u1'),('val2','<u1')])
    
    mframe = [0] * 7
    timestamp = np.zeros(7,dtype=np.uint32)
    dlen = [-1] * 6
    idx = [-1] * 6
    numstatus = [0] * 6
    ovf = [0] * 6
    row = [-1] * 6
    felv1=-1
    tlu = 0
    tlu_flg=0
    start=0
    n = 10000000

    with tables.open_file(fin) as tb:
        end=int(len(tb.root.raw_data))
        print "# of raw data",end
        t0 = time.time()
        dat = np.empty(n, dtype=m26_hit_dtype)
        dat = dat.view(np.recarray)
        with tables.open_file(fout, 'w') as out_file_h5:
            while True:
                tmpend=min(start+n,end)
                hit_dat,raw_i,err =_m26_interpreter(tb.root.raw_data[start:tmpend],dat, idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf,tlu_flg)
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
                    print err,start,raw_i,hex(tb.root.raw_data[start+raw_i])
                    if debug==2:
                        print_start=max(start+raw_i-300,0)
                        for j in range(print_start,start+raw_i+100,1):
                            print "ERROR %4d %4d"%(j-start+raw_i,j),hex(tb.root.raw_data[j])
                        break
                    raw_i=raw_i+1
                if start==0:
                    description = np.zeros((1, ), dtype=m26_hit_dtype).dtype
                    hit_table = out_file_h5.create_table(out_file_h5.root, 
                        name='Hits', 
                        description=description, 
                        title='hit_data')
                hit_table.append(hit_dat)
                hit_table.flush()
                start=start+raw_i
                if start>=end:
                    break
            
if __name__=="__main__":
    import os,sys
    fin=sys.argv[1]
    fout=fin[:-3]+"_hits.h5"
    m26_interpreter(fin,fout)
            


            
    
