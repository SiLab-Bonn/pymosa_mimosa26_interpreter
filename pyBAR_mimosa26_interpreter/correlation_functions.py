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
            if m26_trigger_begin <=m26_trigger_end: ### normal case
                ## TODO check if this covers all the cases
                if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                elif ((m26_trigger_end-fe_trigger) & 0x7FFF )>0x4000:
                    break
            else:  ### overflow of trigger number
                if (fe_trigger >= m26_trigger_begin and fe_trigger <= 0x7fff) or (fe_trigger <= m26_trigger_end and fe_trigger >= 0):        
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                elif ((m26_trigger_end-fe_trigger) & 0x7FFF )>0x4000:
                    break
        if fe_i==fe_data.shape[0]-1:
            #print 'end of fe_data'
            return  fe_index, m26_index

        for m26_i in range(m26_index,m26_data.shape[0]):
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
        if m26_i== m26_data.shape[0]-1:
            #print 'end of m26_data'
            return fe_index, m26_index
        
        #print "go next frame"
        fe_index=fe_prev_break
        fe_prev_break=fe_i
        m26_index=m26_i
    #print "EEROR this should not happen"                    
    return -1, -1

          
