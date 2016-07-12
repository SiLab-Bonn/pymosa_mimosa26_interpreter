from numba import njit
import numpy as np

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

@njit
def correlate_fm(fe_data, m26_data, corr_col, corr_row, dut1, dut2):
    #initialise variables
    #fei4
    fe_index = 0
    fe_prev_break = 0
    fe_buf_index = 0
    fe_trigger = 0
    fe_buf_col = np.zeros(1000, dtype=np.uint32) #make buffer for column correlation
    fe_buf_row = np.zeros(1000, dtype=np.uint32) #make buffer for row correlation
    #m26
    m26_index = 0
    #end of initialisation
    
    while m26_index < m26_data.shape[0]:
        
        m26_frame = m26_data[m26_index]['frame']
        m26_trigger_begin = m26_data[m26_index]['trigger_number_begin']
        m26_trigger_end = m26_data[m26_index]['trigger_number_end']
        
        ##search trigger number in FE data
        fe_buf_index=0
        
        for fe_i in range(fe_index, fe_data.shape[0]):    
                
            fe_trigger = fe_data[fe_i]['trigger_number'] & 0xFFFF
            
            if m26_trigger_begin <= m26_trigger_end: # normal case
                
                # TODO check if this covers all the cases ==============> DOES NOT COVER ALL CASES!!!
                if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                        
                elif ((m26_trigger_end - fe_trigger) & 0x7FFF ) > 0x4000: #0x1000 #if fe_trigger is close to overflow and m26_trigger_end is small, we dont want to break because this fe_data belongs to this m26_data
                    break
            
            else:  # overflow of m26 trigger end
                 
                if (fe_trigger >= m26_trigger_begin and fe_trigger <= 0x7FFF) or (fe_trigger <= m26_trigger_end and fe_trigger >= 0):        
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                
                elif ((m26_trigger_end-fe_trigger) & 0x7FFF ) > 0x4000:  #0x1000 #if fe_trigger is close to overflow and m26_trigger_end is small, we dont want to break because this fe_data belongs to this m26_data
                    break
                    
        if fe_i==fe_data.shape[0] - 1: #end of fe_data, in this case we cant finish merging fe_data to m26_data; add new data to buffer and start from previous index 
            return  fe_index, m26_index

        for m26_i in range(m26_index, m26_data.shape[0]):
            
            if m26_frame == m26_data[m26_i]['frame']:
                
                for i in range(fe_buf_index): #fill histogramms
                    if dut1 == 0:
                        corr_col[fe_buf_row[i], m26_data[m26_i]['column']] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                        corr_row[fe_buf_col[i], m26_data[m26_i]['row']] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
                    elif dut2 == 0:
                        corr_col[m26_data[m26_i]['column'],fe_buf_row[i]] += 1 #m26_col corresponds to fe_row because of geometry of telescope
                        corr_row[m26_data[m26_i]['row'],fe_buf_col[i]] += 1 #m26_row corresponds to fe_col because of geometry of telescope    
            else:
                break
                
        if m26_i== m26_data.shape[0]-1: #end of m26_data, in this case m26_data is to short for corresponding fe_data; add new data to fe_buffer and start from previous fe_index; start from m26_i because needed m26_data < m26_i is already in histogramm 
            #m26_index = m26_i
            return fe_index, m26_i
        
        fe_index=fe_prev_break
        fe_prev_break=fe_i
        m26_index=m26_i
                      
    return -1, -1 # error, this should not happen  


# @njit
# def correlate_mm(m0_data, m1_data, corr_col, corr_row):
#     #initialise variables
#     m0_index = 0
#     m0_prev_i = 0
#     m1_index = 0
#     m1_prev_i = 0
#     m0_i = 0
#     m1_i = 0
#     m0_buf_i = 0
#     m1_buf_i = 0
#     m0_buf_col = np.zeros(1000, dtype=np.uint32)
#     m0_buf_row = np.zeros(1000, dtype=np.uint32)
#     m1_buf_col = np.zeros(1000, dtype=np.uint32)
#     m1_buf_row = np.zeros(1000, dtype=np.uint32)
#     #end of initialisation
#                                                                             #needs to be m_data.shape[0]-1 because indices start from 0; range(m_data.shape[0]) goes from 0 to m_data.shape[0]-1
#     while m0_index < m0_data.shape[0] - 1 and m1_index < m1_data.shape[0] - 1: #need to check both because checking for equality of frames in next if statement; if one data has no matching frames with other data we need to return corresponding indices 
#         
#         #get first frame numbers of both data streams
#         m0_frame = m0_data[m0_index]['frame']
#         m1_frame = m1_data[m1_index]['frame']
#         
#         #check whether frames are equal; if not, increase one of the indices m0_index or m1_index and continue until frames are equal; if no matching frames, loop terminates and returns indices
#         if m0_frame != m1_frame: #frame number can stay the same for several increased indices since mimosa planes can have several hits per frame
#             if m0_frame < m1_frame:
#                 m0_index += 1
#                 continue
#             elif m0_frame > m1_frame: 
#                 m1_index += 1
#                 continue
#                 
#         else: #m0_frame and m1_frame are equal
#                         
#             m0_buf_i = 0 #reset buffer index to overwrite entries in buffer
#             for m0_i in range(m0_index, m0_data.shape[0]): #search m0_data
#                 
#                 if m0_frame == m0_data[m0_i]['frame']: #as long as frames are equal, write to buffer
#                     
#                     m0_buf_col[m0_buf_i] = m0_data[m0_i]['column']
#                     m0_buf_row[m0_buf_i] = m0_data[m0_i]['row']
#                     m0_buf_i += 1
#                 
#                 elif m0_frame < m0_data[m0_i]['frame']:
#                     m0_prev_i = m0_index #index we need if we reach end of m0_data; since we didnt add anything to histogramm yet, next m1_data has to start from here
#                     m0_index = m0_i
#                     break
#             
#             if m0_i == m0_data.shape[0] -1:
#                 return m0_prev_i, m1_index #m0_data finished, next m0_data stream must start from loop index where we started before end of data was reached
# 
#             
#             m1_buf_i = 0 #reset buffer index to overwrite entries in buffer
#             for m1_i in range(m1_index, m1_data.shape[0]): #search m1_data
#                 
#                 if m1_frame == m1_data[m1_i]['frame']: #as long as frames are equal, write to buffer
#                     
#                     m1_buf_col[m1_buf_i] = m1_data[m1_i]['column']
#                     m1_buf_row[m1_buf_i] = m1_data[m1_i]['row']
#                     m1_buf_i += 1
#                 
#                 elif m1_frame < m1_data[m1_i]['frame']: #
#                     m1_prev_i = m1_index #index we need if we reach end of m1_data; since we didnt add anything to histogramm yet, next m1_data has to start from here
#                     m1_index = m1_i
#                     break
#                     
#             if m1_i == m1_data.shape[0] - 1:
#                 return m0_index, m1_prev_i #m1_data finished, next m1_data stream must start from loop index where we started before end of data was reached
#             
#             #fill histogramms
#             for i in range(m0_buf_i):
#                 for j in range(m1_buf_i):
#                     corr_col[m0_buf_col[i]][m1_buf_col[j]] += 1 
#                     corr_row[m0_buf_row[i]][m1_buf_row[j]] += 1
#                     
#     return m0_index, m1_index #only occurs if incoming data streams have no frame numbers in common
    
    
    
@njit
def correlate_mm(m0_data, m1_data, corr_col, corr_row):
    #variables
    m0_index = 0
    m1_index = 0
    #end
    
    if m0_data.shape[0] == 0 or m1_data.shape[0] == 0:
        return m0_index, m1_index
    
    else:
        
        for m0_index in range(m0_data.shape[0]):
            
            m0_frame = m0_data[m0_index]['frame']

            while m1_index < m1_data.shape[0] - 1 and m1_data[m1_index]['frame'] < m0_frame: #keep frame up with outer frame
                m1_index += 1
            
            if m0_index == m0_data.shape[0] - 1 or m1_index == m1_data.shape[0] - 1: #return here if on of the data streams ends, so no correlation for current indices; add this data to next data and then correlate 
                return m0_index, m1_index
            
            for m1_i in range(m1_index, m1_data.shape[0]):
                
                m1_frame = m1_data[m1_i]['frame']
                
                if m1_i == m1_data.shape[0]-1 and m0_frame == m1_frame: #if we reach end of m1_data and frame numbers should still be correlated, return and add this data to next data stream
                    return m0_index, m1_i
                    
                #if frames are equal, fill histogramms
                
                if m0_frame == m1_frame:
                    corr_col[m0_data[m0_index]['column'], m1_data[m1_i]['column']] += 1
                    corr_row[m0_data[m0_index]['row'], m1_data[m1_i]['row']] += 1
                    
                else:
                    break
             
        return -1, -1 #error, should not happen since we return in outer for-loop if one of the indices is m_data.shape[0] -1 

 
@njit
def correlate_ff(f0_data, corr_col, corr_row): #f0_data == f1_data for m26 telescope, just to see something when you select both DUTs as FEI4

    for i in range(f0_data.shape[0]):
        corr_col[f0_data[i]['column']][f0_data[i]['column']] += 1 
        corr_row[f0_data[i]['row']][f0_data[i]['row']] += 1
    return f0_data.shape[0] - 1
     
