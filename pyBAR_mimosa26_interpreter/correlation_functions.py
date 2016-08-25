from numba import njit
import numpy as np
import time

m26_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

@njit
def correlate_fm(fe_data, m26_data, corr_col, corr_row, dut1, dut2, transpose = True):
    '''
    Main function to correlate fe to mimosa data. Correlates data on trigger number, where fe_trigger is assigned to a trigger range of mimosa frames.
    Parameters
    ----------
    fe_data: fe hit data of type 'numpy.ndarray' with data type of fe data
    m26_data: mimosa hit data of type 'numpy.ndarray' with data type of mimosa data
    corr_col: array to store column histogramm of type 'numpy.ndarray' with the correct shape=(column,column)
    corr_row: array to store row histogramm of type 'numpy.ndarray' with the correct shape=(row,row)
    dut1: integer; index of active dut 1, needed to determine which dut is the frontend and correlation order;  frontend to mimosa or mimosa to frontend
    dut2: integer; index of active dut 2, needed to determine which dut is the frontend and correlation order;  frontend to mimosa or mimosa to frontend
    transpose: boolean; if True the fe columns/rows correspond to mimosas rows/columns, if False; fe columns/rows to mimosa columns/rows 
    
    Returns
    -------
    fe_index: int index of m0_data where correlation stops. Data in data buffer below that index will be deleted, from will be kept. Next incoming data will be added to data buffer (dict), starting from m0_index (sth. like buffer[m0] = buffer[m0][m0_index: ].append(m0_data_new) )
    m26_index: int index of m1_data where correlation stops. Data in data buffer below that index will be deleted, from will be kept. Next incoming data will be added to data buffer (dict), starting from m1_index (sth. like buffer[m1] = buffer[m1][m1_index: ].append(m1_data_new) )
    '''
    
    ### initialise variables
    #fei4
    fe_start = 0
    fe_index = 0
    fe_buf_index = 0
    fe_buf_col = np.zeros(1000, dtype=np.uint32) #make buffer for column correlation
    fe_buf_row = np.zeros(1000, dtype=np.uint32) #make buffer for row correlation
    #m26
    m26_index = 0
    ### end of initialisation
    
    ### main loop
    while m26_index < m26_data.shape[0]:
        
        if m26_data[m26_index]['trigger_number_begin'] != 0xFFFF and m26_data[m26_index]['trigger_number_end'] != 0xFFFF: # Mimosa frames without a trigger will have the trigger number 0xFFFF; we want to skip these
            m26_frame = m26_data[m26_index]['frame']
            m26_trigger_begin = m26_data[m26_index]['trigger_number_begin']
            m26_trigger_end = m26_data[m26_index]['trigger_number_end']
        else:
            if m26_index == m26_data.shape[0]-1: # skip as long as current data index is not last element of data array, if last, return
                return fe_index, m26_index
            else:
                m26_index += 1 # skip here
                continue
            
        while fe_data[fe_index]['trigger_number'] & 0x7FFF < m26_trigger_begin: # keep the fe trigger up with the beginning of the m26 trigger range to set fe_index to starting trigger
                if fe_index == fe_data.shape[0]-1:
                    return fe_index, m26_index
                elif fe_data[fe_index]['trigger_number'] & 0x7FFF == m26_trigger_begin: # break if fe_index matches the right trigger
                    break
                else:
                    fe_index += 1
                    
        if fe_data[fe_index]['trigger_number'] & 0x7FFF > m26_trigger_begin: # keep up trigger range of mimosa with fe trigger
            if m26_index == m26_data.shape[0]-1:
                return fe_index, m26_index
            else:
                m26_index += 1
                continue
        
        
        ###search trigger number in  fe_data
        fe_buf_index=0 # overwrite buffer
        for fe_i in range(fe_index, fe_data.shape[0]):    
                
            fe_trigger = fe_data[fe_i]['trigger_number'] & 0x7FFF # get trigger number of current index fe_i

            if m26_trigger_begin <= m26_trigger_end: # normal case, no trigger overflow
                
                if fe_trigger >= m26_trigger_begin and fe_trigger <= m26_trigger_end:
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                        
                elif ((m26_trigger_end - fe_trigger) & 0x7FFF ) > 0x4000: #if fe_trigger is close to overflow and m26_trigger_end is small, we dont want to break because this fe_data belongs to this m26_data
                    break
            
            else:  # overflow of m26 trigger end
                
                if (fe_trigger >= m26_trigger_begin and fe_trigger <= 0x7FFF) or (fe_trigger <= m26_trigger_end and fe_trigger >= 0):        
                        fe_buf_col[fe_buf_index] = fe_data[fe_i]['column']
                        fe_buf_row[fe_buf_index] = fe_data[fe_i]['row'] 
                        fe_buf_index += 1
                
                elif ((m26_trigger_end-fe_trigger) & 0x7FFF ) > 0x4000:  #if fe_trigger is close to overflow and m26_trigger_end is small, we dont want to break because this fe_data belongs to this m26_data
                    break
        
        if fe_i == fe_data.shape[0] - 1: #end of fe_data, in this case we cant finish merging fe_data to m26_data; add new data to buffer and start from previous index 
            return  fe_index, m26_index
        
        for m26_i in range(m26_index, m26_data.shape[0]):
            
            if m26_frame == m26_data[m26_i]['frame']:
                
                for i in range(fe_buf_index): # fill histogramms
                    
                    if transpose == True: # m26_col corresponds to fe_row and m26_row corresponds to fe_col because of our geometry of our telescope
                        
                        if dut1 == 0: # correlate fe to m26
                            corr_col[fe_buf_row[i], m26_data[m26_i]['column']] += 1 
                            corr_row[fe_buf_col[i], m26_data[m26_i]['row']] += 1    
                        elif dut2 == 0: # correlate m26 to fe
                            corr_col[m26_data[m26_i]['column'],fe_buf_row[i]] += 1 
                            corr_row[m26_data[m26_i]['row'],fe_buf_col[i]] += 1 
                    
                    else: #m26_col corresponds to fe_col and m26_row corresponds to fe_row
                        
                        if dut1 == 0: # correlate fe to m26
                            corr_col[fe_buf_col[i], m26_data[m26_i]['column']] += 1 
                            corr_row[fe_buf_row[i], m26_data[m26_i]['row']] += 1     
                        elif dut2 == 0: # correlate m26 to fe
                            corr_col[m26_data[m26_i]['column'],fe_buf_col[i]] += 1 
                            corr_row[m26_data[m26_i]['row'],fe_buf_row[i]] += 1 
            else:
                break
                
        if m26_i == m26_data.shape[0]-1: #end of m26_data, in this case m26_data is to short for corresponding fe_data; add new data to fe_buffer and start from previous fe_index; start from m26_i because needed m26_data < m26_i is already in histogramm 
            return fe_index, m26_i
        
        fe_index=fe_start
        fe_start=fe_i
        m26_index=m26_i
                      
    return -1, -1 # error, this should not happen
    
    
    
@njit
def correlate_fm_beta(fe_data, m26_data, corr_col, corr_row, dut1, dut2, transpose = True): #beta; do not consider trigger overflow
    #initialise variables
    #fei4
    fe_index = 0 # index
    fe_start = 0 # starting index
    fe_check = 0 # index to check whether correlation is possible within mimosa trigger range for current fe data
    #m26
    m26_index = 0 # index
    #end of initialisation
    
    if m26_data.shape[0] == 0 or fe_data.shape[0] == 0: # skip empty data
        return fe_start, m26_index
    
    while m26_index < m26_data.shape[0]: # main loop
        
        m26_frame = m26_data[m26_index]['frame'] # get mimosas frame number
        
        for m26_i in range(m26_index, m26_data.shape[0]):
            
            if m26_frame != m26_data[m26_i]['frame']: # do not correlate if frames are not equal
                break
            
            m26_trigger_begin  = m26_data[m26_i]['trigger_number_begin'] # get mimosa trigger range
            m26_trigger_end = m26_data[m26_i]['trigger_number_end']
            
            
            if m26_trigger_begin == 0xFFFF or m26_trigger_end == 0xFFFF or m26_trigger_begin == 0 or m26_trigger_end == 0: # if mimosa frame has no trigger, trigger begin and end will be 0xFFFF; skip since we can not correlate; also skip 0 since we do not ant to deal with overflow
                if m26_i == m26_data.shape[0]-1:
                    return fe_start, m26_i
                else:
                    continue
            
            if m26_trigger_begin > m26_trigger_end: # trigger overflow will be skipped here
                if m26_i == m26_data.shape[0]-1:
                    return fe_start, m26_i
                else:
                    continue
            
            
            while fe_data[fe_start]['trigger_number'] & 0x7FFF < m26_trigger_begin: # keep up fe trigger with mimosa trigger range
                if fe_start == fe_data.shape[0]-1:
                    return fe_start, m26_i
                elif fe_data[fe_start]['trigger_number'] & 0x7FFF == m26_trigger_begin:
                    break
                else:
                    fe_start += 1
                    
            if fe_data[fe_start]['trigger_number'] & 0x7FFF > m26_trigger_begin: # keep up mimosa trigger range with fe trigger
                if m26_i == m26_data.shape[0]-1:
                    return fe_start, m26_i 
                else:
                    continue
                    
            fe_index = fe_start # set fe_index to correct starting index
            fe_check = fe_start # set fe_check to correct starting index
            
            while fe_data[fe_check]['trigger_number'] & 0x7FFF >= m26_trigger_begin and fe_data[fe_check]['trigger_number'] & 0x7FFF <= m26_trigger_end: # check whether correlation for current mimosa trigger range can be done within length of fe data; if not return; do this because we directly correlate and do not buffer
                if fe_check == fe_data.shape[0]-1:
                    return fe_start, m26_i
                else:
                    fe_check += 1

                    
            while fe_data[fe_index]['trigger_number'] & 0x7FFF >= m26_trigger_begin and fe_data[fe_index]['trigger_number'] & 0x7FFF <= m26_trigger_end: # correlate
                
                if transpose == True: # m26_col corresponds to fe_row and m26_row corresponds to fe_col because of our geometry of our telescope
                    
                    if dut1 == 0: # correlate fe to m26
                        corr_col[fe_data[fe_index]['row'], m26_data[m26_i]['column']] += 1 
                        corr_row[fe_data[fe_index]['column'], m26_data[m26_i]['row']] += 1    
                    elif dut2 == 0: # correlate m26 to fe
                        corr_col[m26_data[m26_i]['column'],fe_data[fe_index]['row']] += 1 
                        corr_row[m26_data[m26_i]['row'],fe_data[fe_index]['column']] += 1 
                
                else: #m26_col corresponds to fe_col and m26_row corresponds to fe_row
                    
                    if dut1 == 0: # correlate fe to m26
                        corr_col[fe_data[fe_index]['column'], m26_data[m26_i]['column']] += 1 
                        corr_row[fe_data[fe_index]['row'], m26_data[m26_i]['row']] += 1     
                    elif dut2 == 0: # correlate m26 to fe
                        corr_col[m26_data[m26_i]['column'],fe_data[fe_index]['column']] += 1 
                        corr_row[m26_data[m26_i]['row'],fe_data[fe_index]['row']] += 1 
                
                fe_index += 1
        
        if m26_i == m26_data.shape[0]-1: # if end of mimosa data is reached
            return fe_start, m26_i
        
        m26_index = m26_i # set new starting point
            
    return -1, -1 
        
                
                        
                
                        
                    
    
        



@njit
def correlate_mm(m0_data, m1_data, corr_col, corr_row):
    '''
    Main function to correlate mimosa to mimosa data. Correlates mimosa data on frame number, where all permutations are concidered.
    Parameters
    ----------
    m0_data: mimosa hit data of type 'numpy.ndarray' with data type of mimosa data
    m1_data: mimosa hit data of type 'numpy.ndarray' with data type of mimosa data
    corr_col: array to store column histogramm of type 'numpy.ndarray' with the correct shape=(column,column)
    corr_row: array to store row histogramm of type 'numpy.ndarray' with the correct shape=(row,row)
    
    Returns
    -------
    m0_index: int index of m0_data where correlation stops. Data in data buffer below that index will be deleted, from will be kept. Next incoming data will be added to data buffer (dict), starting from m0_index (sth. like buffer[m0] = buffer[m0][m0_index: ].append(m0_data_new) )
    m1_index: int index of m1_data where correlation stops. Data in data buffer below that index will be deleted, from will be kept. Next incoming data will be added to data buffer (dict), starting from m1_index (sth. like buffer[m1] = buffer[m1][m1_index: ].append(m1_data_new) )
    '''
    
    #declare variables
    m0_index = 0
    m1_index = 0
    #end
    
    if m0_data.shape[0] == 0 or m1_data.shape[0] == 0: # if one mimosa data is empty, return and keep both data in buffer
        return m0_index, m1_index
    
    else:
        
        for m0_index in range(m0_data.shape[0]): # go trough first mimosa data m0_data
            
            m0_frame = m0_data[m0_index]['frame'] # get the frame number corresponding to current m0_index

            while m1_index < m1_data.shape[0] - 1 and m1_data[m1_index]['frame'] < m0_frame: #keep m1_frame up with outer m0_frame
                m1_index += 1
            
            if m0_index == m0_data.shape[0] - 1 or m1_index == m1_data.shape[0] - 1: #return here if on of the data streams ends, so no correlation for current indices; add this data to next data and then correlate 
                return m0_index, m1_index
            
            for m1_i in range(m1_index, m1_data.shape[0]): # go trough second mimosa data m1_data, start from m1_index which was kept up so that you start from same frame
                
                m1_frame = m1_data[m1_i]['frame'] # get the frame number corresponding to current m1_i
                
                if m1_i == m1_data.shape[0]-1 and m0_frame == m1_frame: #if we reach end of m1_data and frame numbers are equal, return and add this data to next data stream; we dont know if next data's frames need to be correlated to this data's last frame
                    return m0_index, m1_i
                
                if m0_frame == m1_frame: # if frames are equal, fill histogramms
                    
                    corr_col[m0_data[m0_index]['column'], m1_data[m1_i]['column']] += 1
                    corr_row[m0_data[m0_index]['row'], m1_data[m1_i]['row']] += 1
        
                else:
                    break
             
        return -1, -1 #error, should not happen since we return in outer for-loop if one of the indices is m_data.shape[0] -1 

 
@njit
def correlate_ff(f0_data, corr_col, corr_row): #f0_data == f1_data for m26 telescope, just to see something when you select both DUTs as FEI4
    '''
    Main function to correlate fei4 data to itself. Takes only one fei4 data input, because in m26 telescope we only have one fe reference plane. Just loops over the data and adds to histogramm
    Parameters
    ----------
    f0_data: fe hit data of type 'numpy.ndarray' with data type of fei4 data, will be correlated to itself
    corr_col: array to store column histogramm of type 'numpy.ndarray' with the correct shape=(column,column)
    corr_row: array to store row histogramm of type 'numpy.ndarray' with the correct shape=(row,row)
    
    Returns
    -------
    int which is equal to the length of input data. Since data is equal, everything can be correlated and nothing has to stay in data buffer
    '''
    for i in range(f0_data.shape[0]): # go trough fei4 data and immidiately histogramm
        corr_col[f0_data[i]['column']][f0_data[i]['column']] += 1 
        corr_row[f0_data[i]['row']][f0_data[i]['row']] += 1
    return f0_data.shape[0] - 1
     
