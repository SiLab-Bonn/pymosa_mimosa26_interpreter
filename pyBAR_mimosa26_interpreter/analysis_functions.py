from numba import njit
import numpy as np

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])
#@njit
def build_corr_fm(fe_hits, m26_hits, corr_x, corr_y):
    fh_i=0    
    fe_index=0
    fe_index_previous=0
    fh_t=0
    mh_i=0
    mh_i1=0
    m26_index=0
    m26_trigger_end=0
    m26_buf_i=0
    m26_bufx=np.empty(300,dtype=type(1))
    m26_bufy=np.empty(300,dtype=type(1))
    fe_buf_i=0
    fe_bufx=np.empty(300,dtype=type(1))
    fe_bufy=np.empty(300,dtype=type(1))
    while m26_index<m26_hits.shape[0]:
        if m26_hits[m26_index]["trigger_number_begin"]==0:
            m26_index=m26_index+1
            continue
        #print "m26 hit",m26_index,m26_hits[m26_index]
        ### search fe data
        if m26_hits[m26_index]["trigger_number_begin"] > m26_hits[m26_index]["trigger_number_end"]:
            trig_flag=1
        else:
            trig_flag=0
        #print "fe start", fe_buf_i,m26_buf_i
        for fh_i in range(fe_index,fe_hits.shape[0]):
            fh_t=fe_hits[fh_i]["trigger_number"] & 0xFFFF
            if fh_i != 0:
                fh_t_prev = fe_hits[fh_i-1]["trigger_number"] & 0xFFFF
            else:
                fh_t_prev=fh_t
            if trig_flag==0:
		        if fh_t>=m26_hits[m26_index]["trigger_number_begin"] \
		          and fh_t<=m26_hits[m26_index]["trigger_number_end"]:
		            fe_bufx[fe_buf_i]=fe_hits[fh_i]["column"]
		            fe_bufy[fe_buf_i]=fe_hits[fh_i]["row"]
		            fe_buf_i=fe_buf_i+1
		            fe_index=fh_i
		        elif fh_t > m26_hits[m26_index]["trigger_number_end"]:
		            fe_index_previous = fe_index
		            fe_index=fh_i
		            break
		        elif abs(fh_t_prev-fh_t) > 0x4000:
		            break
            else: # flg=1
		        if fh_t>=m26_hits[m26_index]["trigger_number_begin"] \
		          or  fh_t<=m26_hits[m26_index]["trigger_number_end"]:
		            fe_bufx[fe_buf_i]=fe_hits[fh_i]["column"]
		            fe_bufy[fe_buf_i]=fe_hits[fh_i]["row"]
		            fe_buf_i=fe_buf_i+1
		            fe_index=fh_i
		        elif fh_t> m26_hits[m26_index]["trigger_number_end"] \
                  and fh_t<0x4000:
		              fe_index_previous = fe_index
		              fe_index=fh_i
		              #print "break begin end",m26_hits[m26_index]["trigger_number_begin"],m26_hits[m26_index]["trigger_number_end"]
		              break
            if fe_buf_i> 300:
                return -1, -1
        if fh_i>=fe_hits.shape[0]-1:
#             print "return by fei4", m26_hits[:m26_index],m26_hits[m26_index]
#             print fe_hits.dtype
#             print fe_hits[:fe_index],fe_hits[fe_index]
            return fe_index,m26_index
        ### search m data
#         print "m26 start", fe_buf_i,m26_buf_i,"+1",m26_hits[m26_index]["frame"]+1,m26_hits[m26_index]["frame"]+2
        for mh_i in range(m26_index,m26_hits.shape[0]): 
            if m26_hits[mh_i]["frame"]==m26_hits[m26_index]["frame"]+1:
                #print "+1",m26_hits[m26_index]["frame"],m26_hits[mh_i]["frame"],m26_buf_i,m26_hits.shape[0],fe_hits.shape[0]
                mh_i1=mh_i
                m26_bufx[m26_buf_i]=m26_hits[mh_i]["column"]
                m26_bufy[m26_buf_i]=m26_hits[mh_i]["row"]
                m26_buf_i=m26_buf_i+1
            elif m26_hits[mh_i]["frame"]==m26_hits[m26_index]["frame"]+2:
                #print "+2",m26_hits[m26_index]["frame"],m26_hits[mh_i]["frame"],m26_buf_i,m26_buf_i,m26_hits.shape[0],fe_hits.shape[0]
                m26_bufx[m26_buf_i]=m26_hits[mh_i]["column"]
                m26_bufy[m26_buf_i]=m26_hits[mh_i]["row"]
                m26_buf_i=m26_buf_i+1
            elif m26_hits[mh_i]["frame"]>m26_hits[m26_index]["frame"]+2:
                m26_index=mh_i1
                #print "break",m26_hits[mh_i]["frame"],m26_hits[m26_index]["frame"],m26_buf_i,m26_hits.shape[0],fe_hits.shape[0]
                break
            if m26_buf_i> 300:
                return -1, -1
        #print "buf start m26x", m26_bufx[:m26_buf_i]
        #print "buf start fex", fe_bufx[:fe_buf_i]
        for i in range(fe_buf_i): 
            for j in range(m26_buf_i):
                corr_x[m26_bufx[j]][fe_bufx[i]]=corr_x[m26_bufx[j]][fe_bufx[i]]+1
                corr_y[m26_bufy[j]][fe_bufy[i]]=corr_y[m26_bufy[j]][fe_bufy[i]]+1
        fe_buf_i=0
        m26_buf_i=0
        if mh_i>=m26_hits.shape[0]-1:
#             print "return by mimosa",m26_index,m26_hits[:m26_index],m26_hits[m26_index]
#             print fe_hits.dtype
#             print fe_index_previous,fe_index,fe_hits[:fe_index],fe_hits[fe_index]
            fe_index = fe_index_previous
#             print 'fe trigger number',
            for ff in fe_hits:
                print ff['trigger_number'] & 0xFFFF,
#             print ""
            return fe_index,m26_index
    #print m26_hits[:m26_index],m26_hits[m26_index]
    #print fe_hits.dtype
    #print fe_hits[:fe_index],fe_hits[fe_index]
    return fe_index,m26_index
@njit
def build_corr_mm(m26_hits0, m26_hits1, corr_x, corr_y):
    idx0=0
    i0=0
    idx1=0
    i1=0
    buf_i0=0
    buf_i1=0
    bufx0=np.empty(300,dtype='<u2')
    bufy0=np.empty(300,dtype='<u4')
    bufx1=np.empty(300,dtype='<u2')
    bufx1=np.empty(300,dtype='<u4')
    while idx0<m26_hits0.shape[0]:
        ### search first plane
        for i0, h0 in enumerate(m26_hits0[idx0:]):
            if h0["frame"]==m26_hits0[idx0]["frame"]:
                bufx0[buf_i0]=fh["col"]
                bufy0[buf_i0]=fh["row"]
                buf_i0=buf_i0+1
            elif h0["frame"]>m26_hits0[idx0]["frame"]:
                  idx0=i0
                  break
        ### search m data
        for i1, h1 in enumerate(m26_hits1[idx1:]):
            if h1["frame"]==m26_hits1[idx1]["frame"]:
                bufx1[buf_i1]=fh["col"]
                bufy1[buf_i1]=fh["row"]
                buf_i1=buf_i1+1
            elif h1["frame"]>m26_hits1[idx1]["frame"]:
                  idx1=i1
                  break
        for i in range(buf_i0):
             for j in range(buf_i1):
                 corr_x[bufx1[j]][bufx0[i]]=corr_x[bufx1[j]][bufx0[i]]+1
                 corr_y[bufy1[j]][bufy0[i]]=corr_y[bufy1[j]][bufy0 [i]]+1
        if i0>=m26_hits0.shape[0]-1:
            break
        if i1>=m26_hits1.shape[0]-1:
            break



