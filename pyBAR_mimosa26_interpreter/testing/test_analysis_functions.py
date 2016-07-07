import sys, os, time

import numpy as np
from pyBAR_mimosa26_interpreter import correlation_functions

###############
### case 1 
fe_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('relative_BCID', 'u1'), ('LVL1ID', '<u2'), ('column', 'u1'), ('row', '<u2'), ('tot', 'u1'), ('BCID', '<u2'), ('TDC', '<u2'), ('TDC_time_stamp', 'u1'), ('trigger_status', 'u1'), ('service_record', '<u4'), ('event_status', '<u2')])

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])
                      #ev    timstam     b      e      p    f          c    r     status
m26_hits = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0)],hit_dtype)
                     #ev    trig       bc lvl c   r    tot 
fe_hits = np.array([(45593, 0xFFF3253, 7, 10, 40, 120, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3254, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3261, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3262, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64)
                    ], fe_dtype)

corr_col= np.zeros((336,1152), dtype=np.uint32) # used to be self.hists_column_corr / empty dict to save every dut with its IP as key and data as value
corr_row = np.zeros((80,576), dtype=np.uint32)  # used to be self.hists_row_corr /

print hex(24326746L),hex(12883)

fe_index, m26_index=correlation_functions.correlate_fm(fe_hits, m26_hits, corr_col, corr_row, 0, 1)
if fe_index==0 and m26_index==0:## and corr_col==xx and corr_row==xx    
    print 'CORRECT!!', 'fe',fe_index,'m26',m26_index
else:
    print "There is a bug in your function!!"
    

###############
### case 2
fe_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('relative_BCID', 'u1'), ('LVL1ID', '<u2'), ('column', 'u1'), ('row', '<u2'), ('tot', 'u1'), ('BCID', '<u2'), ('TDC', '<u2'), ('TDC_time_stamp', 'u1'), ('trigger_status', 'u1'), ('service_record', '<u4'), ('event_status', '<u2')])

hit_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])
                      #ev    timstam     b      e      p    f          c    r     status
m26_hits = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                     (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0)],hit_dtype)
                     #ev    trig       bc lvl c   r    tot 
fe_hits = np.array([(45593, 0xFFF3253, 7, 10, 40, 120, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3254, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3261, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                    (45593, 0xFFF3262, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64)
                    ], fe_dtype)

corr_col= np.zeros((336,1152), dtype=np.uint32) # used to be self.hists_column_corr / empty dict to save every dut with its IP as key and data as value
corr_row = np.zeros((80,576), dtype=np.uint32)  # used to be self.hists_row_corr /

print hex(24326746L),hex(12883)

fe_index, m26_index=correlation_functions.correlate_fm(fe_hits, m26_hits, corr_col, corr_row, 0, 1)
if fe_index==0 and m26_index==0:## and corr_col==xx and corr_row==xx    
    print 'CORRECT!!', 'fe',fe_index,'m26',m26_index
else:
    print "There is a bug in your function!!"