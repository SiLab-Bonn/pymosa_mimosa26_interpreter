from numba import njit
import numpy as np
import time
hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

#@njit
def correlate_fm(fe_data, m26_data, corr_col, corr_row, dut1, dut2):
    #initialise variables
    #fei4
    fe_index1 = 0
    fe_index2 = 0
    fe_buf_index = 0
    fe_hit_index = 0
    fe_trigger = 0
    fe_runtrough = 0
    fe_runflag = 0
    fe_buf_col = np.zeros(1000, dtype=np.uint32) #make buffer for column correlation
    fe_buf_row = np.zeros(1000, dtype=np.uint32) #make buffer for row correlation
    #m26
    m26_index1 = 0
    m26_index2 = 0
    m26_index_prev = 0
    m26_buf_index = 0
    m26_hit_index = 0
    m26_runtrough = 0
    m26_buf_col = np.zeros(1000, dtype=np.uint32) #make buffer for column correlation
    m26_buf_row = np.zeros(1000, dtype=np.uint32) #make buffer for row correlation
    #trigger
    trigger_overflow = 0
    #end of initialisation
    
    
    for m26_index1 in range(m26_data.shape[0]):
        #~ if fe_runflag == 1:
            #~ m26_index1 = m26_index_prev
        
        m26_frame = m26_data[m26_index1]['frame']
        m26_trigger_begin = m26_data[m26_index1]['trigger_number_begin']
        m26_trigger_end = m26_data[m26_index1]['trigger_number_end']
        
        if m26_trigger_begin > m26_trigger_end:
            trigger_overflow = 1
        else:
            trigger_overflow = 0
        
        m26_index2 = m26_index1
        m26_buf_index = 0
        m26_runtrough = 0
        
        while m26_index2 < m26_data.shape[0]:
            
            if m26_data[m26_index2]['frame'] == m26_frame:
                m26_buf_col[m26_buf_index] = m26_data[m26_index2]['column']
                m26_buf_row[m26_buf_index] = m26_data[m26_index2]['row']
                m26_buf_index += 1
                m26_index2 += 1
            
            elif m26_data[m26_index2]['frame'] > m26_frame:
                m26_index1_prev = m26_index1
                m26_index1 = m26_index2
                break
            elif m26_data[m26_index2]['frame'] < m26_frame:
                m26_index2 += 1
                m26_runtrough += 1
        
        fe_index1 = fe_index2
        fe_buf_index = 0
        fe_runtrough = 0
        
        while fe_index1 < fe_data.shape[0]:
            
            fe_trigger = fe_data[fe_index1]['trigger_number'] & 0xFFFF
            
            if fe_trigger < m26_trigger_begin:
                    fe_index1 += 1
                    fe_runtrough += 1
            
            if trigger_overflow == 0:
                
                if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                    
                    fe_buf_col[fe_buf_index] = fe_data[fe_index1]['column']
                    fe_buf_row[fe_buf_index] = fe_data[fe_index1]['row'] 
                    fe_buf_index += 1
                    fe_index1 += 1
                
                elif fe_trigger > m26_trigger_end:
                    fe_prev = fe_index2
                    fe_index2 = fe_index1
                    break
                
                #~ else: #fe_trigger < m26_trigger_begin:
                    #~ fe_index1 += 1
                    #~ fe_runtrough += 1
            
            elif trigger_overflow == 1:
                
                if (fe_trigger >= m26_trigger_begin and fe_trigger <= 0x7fff) or (fe_trigger <= m26_trigger_end and fe_trigger >= 0):
                    
                    fe_buf_col[fe_buf_index] = fe_data[fe_index1]['column']
                    fe_buf_row[fe_buf_index] = fe_data[fe_index1]['row'] 
                    fe_buf_index += 1
                    fe_index1 += 1
                
                elif fe_trigger > m26_trigger_end and fe_trigger < 0x3fff:
                    fe_index_prev = fe_index2
                    fe_index2 = fe_index1
                    break
                
                #~ else: #(fe_trigger < m26_trigger_begin):
                    #~ fe_index1 += 1
                    #~ fe_runtrough += 1
        
        #print "FEINDEX", fe_index1
        
        if fe_runtrough >= fe_data.shape[0]-1:
            #~ fe_runflag = 1
            continue
       
        #print np.max(fe_buf_col), np.max(fe_buf_row)
        #print np.max(m26_buf_col), np.max(m26_buf_row)
        
        for i in range(fe_buf_index): #fill histogramms
            for j in range(m26_buf_index):
                if dut1 == 0:
                    corr_col[fe_buf_row[i]][m26_buf_col[j]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                    corr_row[fe_buf_col[i]][m26_buf_row[j]] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
                elif dut2 == 0:
                    corr_col[m26_buf_col[j]][fe_buf_row[i]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                    corr_row[m26_buf_row[j]][fe_buf_col[i]] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
                        
    return fe_index1, m26_index1
                
