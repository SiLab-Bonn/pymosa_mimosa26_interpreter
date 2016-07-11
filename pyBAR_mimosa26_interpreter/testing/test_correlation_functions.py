import sys,os,time
import numpy as np

#import pyBAR_mimosa26_interpreter
from pyBAR_mimosa26_interpreter import correlation_functions


fe_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('relative_BCID', 'u1'), ('LVL1ID', '<u2'), ('column', 'u1'), ('row', '<u2'), ('tot', 'u1'), ('BCID', '<u2'), ('TDC', '<u2'), ('TDC_time_stamp', 'u1'), ('trigger_status', 'u1'), ('service_record', '<u4'), ('event_status', '<u2')])

m26_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

corr_col_mm= np.zeros((1152,1152), dtype=np.uint32) 
corr_row_mm = np.zeros((576,576), dtype=np.uint32) 
#test correlate_mm

#case1: m0_data == m1_data
                     
m0_data = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                    (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                    (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                    (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],m26_dtype)

m1_data = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                    (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                    (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                    (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],m26_dtype)
#case2a: m0_data has range of frame numbers and every frame of m1_data is in that range #case2b: other way round
m0_data1 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],m26_dtype)

m1_data1 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],m26_dtype)
#case3a: m0_data's frame numbers are all bigger than m_data's #case3b: other way round
m0_data2 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],m26_dtype)

m1_data2 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435468L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435468L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0)],m26_dtype)
#case4a: m0_datais empty, m1_data not #case3b: other way round
m0_data3 = np.array([],m26_dtype)

m1_data3 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435468L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435468L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0)],m26_dtype)
#case 5a:
m0_data4 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435470L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435475L, 555, 333L, 52171, 0)],m26_dtype)

m1_data4 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],m26_dtype)

m0_index, m1_index = correlation_functions.correlate_mm(m0_data, m1_data, corr_col_mm,corr_row_mm)
m0_index1a, m1_index1a = correlation_functions.correlate_mm(m0_data1, m1_data1, corr_col_mm,corr_row_mm)
m0_index1b, m1_index1b = correlation_functions.correlate_mm(m1_data1, m0_data1, corr_col_mm,corr_row_mm)
m0_index2a, m1_index2a = correlation_functions.correlate_mm(m0_data2, m1_data2, corr_col_mm,corr_row_mm)
m0_index2b, m1_index2b = correlation_functions.correlate_mm(m1_data2, m0_data2, corr_col_mm,corr_row_mm)
m0_index3a, m1_index3a = correlation_functions.correlate_mm(m0_data3, m1_data3, corr_col_mm,corr_row_mm)
m0_index3b, m1_index3b = correlation_functions.correlate_mm(m1_data3, m0_data3, corr_col_mm,corr_row_mm)



if m0_index == 2 and m1_index == 2:
    print "CORRECT:", m0_index, m1_index
else:
    print "BUG! m0_index/m1_index should be 2/2, is:", m0_index, m1_index

if m0_index1a == 2 and m1_index1a == 0:
    print "CORRECT:",m0_index1a, m1_index1a
else:
    print "BUG! m0_index/m1_index should be 2/0, is:", m0_index1a, m1_index1a

if m0_index1b == 0 and m1_index1b == 0:
    print "CORRECT:", m0_index1b, m1_index1b
else:
    print "BUG! m0_index/m1_index should be 0/0, is:", m0_index1b, m1_index1b

if m0_index2a == 0 and m1_index2a == 3:
    print "CORRECT:",m0_index2a, m1_index2a
else:
    print "BUG! m0_index/m1_index should be 0/3, is:", m0_index2a, m1_index2a

if m0_index2b == 3 and m1_index2b == 0:
    print "CORRECT:", m0_index2b, m1_index2b
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index2b, m1_index2b
    
if m0_index3a == 0 and m1_index3a == 0:
    print "CORRECT:", m0_index3a, m1_index3a
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index3a, m1_index3a

if m0_index3b == 0 and m1_index3b == 0:
    print "CORRECT:", m0_index3b, m1_index3b
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index3b, m1_index3b
print "FAAAAAAAAAAAST\n"
m0_index, m1_index = correlation_functions.correlate_mm_fast(m0_data, m1_data, corr_col_mm,corr_row_mm)
m0_index1a, m1_index1a = correlation_functions.correlate_mm_fast(m0_data1, m1_data1, corr_col_mm,corr_row_mm)
m0_index1b, m1_index1b = correlation_functions.correlate_mm_fast(m1_data1, m0_data1, corr_col_mm,corr_row_mm)
m0_index2a, m1_index2a = correlation_functions.correlate_mm_fast(m0_data2, m1_data2, corr_col_mm,corr_row_mm)
m0_index2b, m1_index2b = correlation_functions.correlate_mm_fast(m1_data2, m0_data2, corr_col_mm,corr_row_mm)
m0_index3a, m1_index3a = correlation_functions.correlate_mm_fast(m0_data3, m1_data3, corr_col_mm,corr_row_mm)
m0_index3b, m1_index3b = correlation_functions.correlate_mm_fast(m1_data3, m0_data3, corr_col_mm,corr_row_mm)
m0_index4a, m1_index4a = correlation_functions.correlate_mm_fast(m0_data4, m1_data4, corr_col_mm,corr_row_mm)
m0_index4b, m1_index4b = correlation_functions.correlate_mm_fast(m1_data4, m0_data4, corr_col_mm,corr_row_mm)


if m0_index == 3 and m1_index == 3:
    print "CORRECT:", m0_index, m1_index
else:
    print "BUG! m0_index/m1_index should be 3/3, is:", m0_index, m1_index

if m0_index1a == 2 and m1_index1a == 3:
    print "CORRECT:",m0_index1a, m1_index1a
else:
    print "BUG! m0_index/m1_index should be 2/3, is:", m0_index1a, m1_index1a

if m0_index1b == 3 and m1_index1b == 2:
    print "CORRECT:", m0_index1b, m1_index1b
else:
    print "BUG! m0_index/m1_index should be 3/2, is:", m0_index1b, m1_index1b

if m0_index2a == 0 and m1_index2a == 3:
    print "CORRECT:",m0_index2a, m1_index2a
else:
    print "BUG! m0_index/m1_index should be 0/3, is:", m0_index2a, m1_index2a

if m0_index2b == 3 and m1_index2b == 0:
    print "CORRECT:", m0_index2b, m1_index2b
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index2b, m1_index2b
    
if m0_index3a == 0 and m1_index3a == 0:
    print "CORRECT:", m0_index3a, m1_index3a
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index3a, m1_index3a

if m0_index3b == 0 and m1_index3b == 0:
    print "CORRECT:", m0_index3b, m1_index3b
else:
    print "BUG! m0_index/m1_index should be 3/0, is:", m0_index3b, m1_index3b
    
if m0_index4a == 3 and m1_index4a == 3:
    print "CORRECT:", m0_index4a, m1_index4a
else:
    print "BUG! m0_index/m1_index should be 3/3, is:", m0_index4a, m1_index4a

if m0_index4b == 3 and m1_index4b == 2:
    print "CORRECT:", m0_index4b, m1_index4b
else:
    print "BUG! m0_index/m1_index should be 3/2, is:", m0_index4b, m1_index4b
    
#test correlate_fm
#TBD
fe_hits = np.array([(45593, 0xFFF3253, 7, 10, 40, 120, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3254, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3261, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3262, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64)
                    ], fe_dtype)



