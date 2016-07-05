from numba import njit
import numpy as np

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

#@njit
def build_corr_fm(fe_hits, m26_hits, corr_col, corr_row, dut1, dut2): #make correlation between frontend and mimosa
    #fei4 variables
    fe_hit_index = 0 #store loop index to loop through fei4 data
    fe_index = 0 #store starting point for loop
    fe_index_prev = 0
    fe_index_now = 0 #store index to update starting point for fei4 loop
    fe_trigger = 0 # store trigger number of fei4 data; format 0xFFFF
    fe_buf_index = 0 #store fei4 buffer index to access histogramms
    fe_buff_col = np.zeros(1000, dtype=np.uint32) #make buffer for column correlation
    fe_buff_row = np.zeros(1000, dtype=np.uint32) #make buffer for row correlation
    #mimosa variables
    m26_hit_index = 0 # see above
    m26_index = 0 #see above
    m26_frame = 0 #store frame number of current mimosa data
    m26_trigger_begin = 0 #store beginning of trigger range of current frame
    m26_trigger_end = 0 #store end of trigger range of current frame
    m26_buf_index = 0 #see above
    m26_buff_col = np.zeros(1000, dtype=np.uint32) #see above
    m26_buff_row = np.zeros(1000, dtype=np.uint32) #see above
    #general
    t_flag = 0
    
    if dut1 != 0 and dut2 != 0: #if none of the DUTs is 
        return
    
    while m26_index < m26_hits.shape[0]:

        m26_trigger_begin = m26_hits[m26_index]["trigger_number_begin"] # get m 26 trigger range
        m26_trigger_end = m26_hits[m26_index]["trigger_number_end"]
        m26_frame = m26_hits[m26_index]['frame']
        
        if m26_trigger_begin == 0xFFFF and m26_trigger_end == 0xFFFF: #if frame has no trigger, skip
            m26_index += 1
            continue
        
        if m26_trigger_begin > m26_trigger_end: #check if the trigger number overflows
            t_flag = 1 #trigger overflow
        else:
            t_flag = 0 #no overflow
        
        m26_buf_index = 0 #overwrite buffer
        
        for m26_hit_index in range(m26_index, m26_hits.shape[0]): # go trough mimosa hits
            
            if m26_hits[m26_hit_index]['frame'] == m26_frame:
                
                m26_buff_col[m26_buf_index] = m26_hits[m26_hit_index]['column']
                m26_buff_row[m26_buf_index] = m26_hits[m26_hit_index]['row']
                m26_buf_index += 1
                m26_index += 1
            
            elif m26_hits[m26_hit_index]['frame'] > m26_frame:
                m26_index = m26_hit_index
                break
#             else:
#                 break
        if m26_buf_index > m26_buff_col.shape[0] or m26_buf_index > m26_buff_row.shape[0]:
            return -1,-1 #buffer index out of buffer range
        
        fe_buf_index = 0 #overwrite buffer
        
        for fe_hit_index in range(fe_index, fe_hits.shape[0]): # go through fei4 hits
            
            fe_trigger = fe_hits[fe_hit_index]["trigger_number"] & 0xFFFF #get trigger number
            
#             if fe_hit_index != 0:
#                 fe_trigger_prev = fe_hits[fe_hit_index-1]["trigger_number"] & 0xFFFF #get previous trigger number
#             else:
#                 fe_trigger_prev = fe_trigger
            
            fe_index = fe_index_now
                
            if t_flag == 0: #no trigger overflow
                
                if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                    
                    fe_buff_col[fe_buf_index] = fe_hits[fe_hit_index]["column"]
                    fe_buff_row[fe_buf_index] = fe_hits[fe_hit_index]["row"]
                    fe_buf_index += 1
                    fe_index_now = fe_hit_index
                      
                elif fe_trigger > m26_trigger_end:
                    fe_index_prev = fe_index
                    fe_index = fe_hit_index
                    break
                
            else: #trigger overflow
                
                if (fe_trigger >= m26_trigger_begin and fe_trigger <= 0x7fff) or (fe_trigger >=0 and fe_trigger <= m26_trigger_end):
                          
                    fe_buff_col[fe_buf_index] = fe_hits[fe_hit_index]["column"]
                    fe_buff_row[fe_buf_index] = fe_hits[fe_hit_index]["row"]
                    fe_buf_index += 1
                    fe_index_now = fe_hit_index 
                
                elif fe_trigger > m26_trigger_end: # and fe_trigger < 0x4000:
                    fe_index_prev = fe_index
                    fe_index = fe_hit_index
                    break
        
        if fe_buf_index > fe_buff_col.shape[0] or fe_buf_index > fe_buff_row.shape[0]:
            return -1,-1 #buffer index out of buffer range
              
        for i in range(fe_buf_index): #fill histogramms
            for j in range(m26_buf_index):
                if dut1 == 0:
                    corr_col[fe_buff_row[i]][m26_buff_col[j]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                    corr_row[fe_buff_col[i]][m26_buff_row[j]] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
                elif dut2 == 0:
                    corr_col[m26_buff_col[j]][fe_buff_row[i]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                    corr_row[m26_buff_row[j]][fe_buff_col[i]] += 1 #m26_row corresponds to fe_col because of geometry of telescope       
        if m26_index >= m26_hits.shape[0]-1:
            fe_index = fe_index_prev
            return fe_index, m26_index
        
    return fe_index, m26_index

#@njit    
def build_corr_mm(m26_hits0,m26_hits1, corr_col, corr_row):
    #variables
    m0_index = 0
    m1_index = 0
    m0_buf_index = 0
    m1_buf_index = 0
    m0_frame = 0
    m1_frame = 0
    m0_hit_index = 0
    m1_hit_index = 0
    m26_buff_col0 = np.zeros(1000, dtype=np.uint32)
    m26_buff_row0 = np.zeros(1000, dtype=np.uint32)
    m26_buff_col1 = np.zeros(1000, dtype=np.uint32)     
    m26_buff_row1 = np.zeros(1000, dtype=np.uint32)
    #data = 0
    
    #data = [m26_hits0.shape[0], m26_hits1.shape[0]]
    
    while m0_index < m26_hits0.shape[0]:
        try:
            m0_frame = m26_hits0[m0_index]['frame']
            m1_frame = m26_hits1[m1_index]['frame']
        except IndexError:
            return m0_index, m1_index
        m0_buf_index = 0
        for m0_hit_index in range(m0_index, m26_hits0.shape[0]):
            
            if m0_frame == m26_hits0[m0_hit_index]['frame']:
                m26_buff_col0[m0_buf_index] = m26_hits0[m0_hit_index]['column'] 
                m26_buff_row0[m0_buf_index] = m26_hits0[m0_hit_index]['row']
                m0_buf_index += 1
                m0_index += 1
            elif m26_hits0[m0_hit_index]['frame'] > m0_frame:
                m0_index = m0_hit_index
                break
            
        m1_buf_index = 0
        for m1_hit_index in range(m1_index, m26_hits1.shape[0]):
            
            if m1_frame == m26_hits1[m1_hit_index]['frame']:
                m26_buff_col1[m1_buf_index] = m26_hits1[m1_hit_index]['column'] 
                m26_buff_row1[m1_buf_index] = m26_hits1[m1_hit_index]['row']
                m1_buf_index += 1
                m1_index += 1
            elif m26_hits1[m1_hit_index]['frame'] > m1_frame:
                m1_index = m1_hit_index
                break
        
        for i in range(m0_buf_index):
            for j in range(m1_buf_index):
                corr_col[m26_buff_col0[i]][m26_buff_col1[j]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                corr_row[m26_buff_row0[i]][m26_buff_row1[j]] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
        if m0_hit_index >= m26_hits0.shape[0] or m1_hit_index >= m26_hits1.shape[0]:
            break
        
    return m0_index, m1_index




