from numba import njit
import numpy as np

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

#@njit
def build_corr_fm(fe_hits, m26_hits, corr_col, corr_row): #make correlation between frontend and mimosa
    #fei4 variables
    fe_hit_index = 0    
    fe_index = 0
    fe_index_now = 0
    fe_trigger = 0
    fe_buf_index = 0
    fe_buff_col = np.zeros(1000, dtype=np.uint32)
    fe_buff_row = np.zeros(1000, dtype=np.uint32)
    #mimosa variables
    m26_hit_index = 0
    m26_index = 0
    m26_trigger_begin = 0
    m26_trigger_end = 0
    m26_buf_index = 0
    m26_buff_col = np.zeros(1000, dtype=np.uint32)
    m26_buff_row = np.zeros(1000, dtype=np.uint32)
    #general
    #trig_flag = 0
    
    
    while m26_index < m26_hits.shape[0]:
        
        m26_trigger_begin = m26_hits[m26_index]["trigger_number_begin"] # get m 26 trigger range
        m26_trigger_end = m26_hits[m26_index]["trigger_number_end"]
        m26_frame = m26_hits[m26_index]['frame']
        
        if m26_trigger_begin == 0xFFFF and m26_trigger_end == 0xFFFF: #if frame has no trigger, skip
            continue
        
        print "m26 trigger: b------e:",m26_trigger_begin, m26_trigger_end, m26_index 
        
        m26_buf_index = 0 #overwrite buffer
        
        for m26_hit_index in range(m26_index, m26_hits.shape[0]):
            
            if m26_hits[m26_hit_index]['frame'] == m26_frame:
                
                m26_buff_col[m26_buf_index] = m26_hits[m26_hit_index]['column']
                m26_buff_row[m26_buf_index] = m26_hits[m26_hit_index]['row']
                m26_buf_index += 1
                m26_index += 1
            
            else:
                break
        
        for fe_hit_index in range(fe_index, fe_hits.shape[0]): # go through fei4 hits
            
            fe_trigger = fe_hits[fe_hit_index]["trigger_number"] & 0xFFFF #get trigger number
            
            print "fe trigger:", fe_trigger
            fe_index = fe_index_now
            if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                
                fe_buff_col[fe_buf_index] = fe_hits[fe_hit_index]["column"]
                fe_buff_row[fe_buf_index] = fe_hits[fe_hit_index]["row"]
                fe_buf_index += 1
                fe_index_now = fe_hit_index
                
            
            elif fe_trigger > m26_trigger_end:
                break
        
            
        for i in range(fe_buf_index):
            for j in range(m26_buf_index):
                corr_col[m26_buff_col[j]][fe_buff_col[i]] += 1
                corr_row[m26_buff_row[j]][fe_buff_row[i]] += 1
        
        fe_buf_index = 0
        m26_buf_index = 0
    return fe_index, m26_index
    
         
            
    
