import sys
import os
import time
import unittest
import numpy as np

from pyBAR_mimosa26_interpreter import correlation_functions

class TestCorrelation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.corr_col_mm= np.zeros((1152,1152), dtype=np.uint32) 
        cls.corr_row_mm = np.zeros((576,576), dtype=np.uint32)
        cls.corr_col_fm= np.zeros((337,1152), dtype=np.uint32) 
        cls.corr_row_fm = np.zeros((81,576), dtype=np.uint32)
        cls.corr_col_mf= np.zeros((1152,337), dtype=np.uint32) 
        cls.corr_row_mf = np.zeros((576,81), dtype=np.uint32)
        
        cls.fe_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('relative_BCID', 'u1'),('LVL1ID', '<u2'),
                             ('column', 'u1'), ('row', '<u2'), ('tot', 'u1'), ('BCID', '<u2'), ('TDC', '<u2'),
                             ('TDC_time_stamp', 'u1'),('trigger_status', 'u1'), ('service_record', '<u4'), ('event_status', '<u2')])

        cls.m26_dtype = np.dtype([('event_number', '<i8'),('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                                  ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])
    @classmethod
    def tearDownClass(cls):
        pass
    
 
#test correlate_mm

    def test_equal_data(self):  # case1: m0_data == m1_data
                 
        m0_data = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                            (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                            (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                            (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],self.m26_dtype)
        
        m1_data = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                            (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                            (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                            (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],self.m26_dtype)
        
        m0_index, m1_index = correlation_functions.correlate_mm(m0_data, m1_data, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index, 3)
        self.assertEqual(m1_index, 3)
        
    def test_data_end_on_same_frame(self):  # case2a: m0_data has range of frame numbers and every frame of m1_data is in that range # case2b: other way round
        
        m0_data1 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],self.m26_dtype)
        
        m1_data1 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],self.m26_dtype)

        m0_index1a, m1_index1a = correlation_functions.correlate_mm(m0_data1, m1_data1, self.corr_col_mm, self.corr_row_mm)

        self.assertEqual(m0_index1a, 2)
        self.assertEqual(m1_index1a, 3)
        
        m0_index1b, m1_index1b = correlation_functions.correlate_mm(m1_data1, m0_data1, self.corr_col_mm, self.corr_row_mm)
               
        self.assertEqual(m0_index1b, 3)
        self.assertEqual(m1_index1b, 2)
        
    def test_no_common_frame(self):  # case3a: m0_data's frame numbers are all bigger than m1_data's # case3b: other way round
        
        m0_data2 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],self.m26_dtype)
        
        m1_data2 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435468L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435468L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0)],self.m26_dtype)
        
        m0_index2a, m1_index2a = correlation_functions.correlate_mm(m0_data2, m1_data2, self.corr_col_mm, self.corr_row_mm)

        self.assertEqual(m0_index2a, 0)
        self.assertEqual(m1_index2a, 3)
        
        m0_index2b, m1_index2b = correlation_functions.correlate_mm(m1_data2, m0_data2, self.corr_col_mm, self.corr_row_mm)

        self.assertEqual(m0_index2b, 3)
        self.assertEqual(m1_index2b, 0)
        
    def test_empty_data(self):  # case4a: m0_data is empty, m1_data not #case3b: other way round
        
        m0_data3 = np.array([],self.m26_dtype)
        
        m1_data3 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435468L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435468L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435469L, 555, 333L, 52171, 0)],self.m26_dtype)


        m0_index3a, m1_index3a = correlation_functions.correlate_mm(m0_data3, m1_data3, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index3a, 0)
        self.assertEqual(m1_index3a, 0)
        
        m0_index3b, m1_index3b = correlation_functions.correlate_mm(m1_data3, m0_data3, self.corr_col_mm, self.corr_row_mm)

        self.assertEqual(m0_index3b, 0)
        self.assertEqual(m1_index3b, 0)
        
    def test_data_skip_frame(self): # #case 5a: m0_data skips frame numbers #case5b: other way round
        
        m0_data4 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435470L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435475L, 555, 333L, 52171, 0)],self.m26_dtype)

        m1_data4 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],self.m26_dtype)

        m0_index4a, m1_index4a = correlation_functions.correlate_mm(m0_data4, m1_data4, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index4a, 2)
        self.assertEqual(m1_index4a, 3)
        
        m0_index4b, m1_index4b = correlation_functions.correlate_mm(m1_data4, m0_data4, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index4b, 3)
        self.assertEqual(m1_index4b, 2)
        
    def test_data_all_frames_in_common(self): #case6a: m1_data's frames are all in m0_data # case6b: other way round
        
        m0_data5 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435473L, 555, 333L, 52171, 0)],self.m26_dtype)

        m1_data5 = np.array([(29123, 130515341L, 0x3253, 0x3258, 2, 35435471L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3253, 0x3258, 2, 35435472L, 555, 333L, 52171, 0)],self.m26_dtype)

        m0_index5a, m1_index5a = correlation_functions.correlate_mm(m0_data5, m1_data5, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index5a, 2)
        self.assertEqual(m1_index5a, 3)
        
        m0_index5b, m1_index5b = correlation_functions.correlate_mm(m1_data5, m0_data5, self.corr_col_mm, self.corr_row_mm)
        
        self.assertEqual(m0_index5b, 3)
        self.assertEqual(m1_index5b, 2)


    
#test correlate_fm

    def test_fe_trig_in_m26_range(self):
        
        fe_data = np.array([(45593, 0xFFF3253, 7, 10, 40, 120, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3254, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3255, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3256, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64)
                            ], self.fe_dtype)
                            
        m26_data = np.array([(29123, 130515341L, 0x3253, 0x3254, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3254, 0x3255, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3254, 0x3257, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3255, 0x3260, 2, 35435473L, 555, 333L, 52171, 0)],
                             self.m26_dtype)
                    
        fe_index, m26_index = correlation_functions.correlate_fm(fe_data, m26_data, self.corr_col_fm,  self.corr_row_fm, 0,1)
        
        self.assertEqual(fe_index, 1)
        self.assertEqual(m26_index, 1)
        
    def test_fe_trig_not_in_m26_range(self):
        
        fe_data = np.array([(45593, 0xFFF3253, 7, 10, 40, 120, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3254, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3255, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64),
                            (45593, 0xFFF3256, 7, 10, 60, 180, 3, 189, 0, 0, 1, 0L, 64)
                            ], self.fe_dtype)
                            
        m26_data = np.array([(29123, 130515341L, 0x3257, 0x3258, 2, 35435470L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3257, 0x3258, 2, 35435471L, 666, 444L, 52171, 0),
                             (29123, 130515341L, 0x3257, 0x3258, 2, 35435472L, 555, 333L, 52171, 0),
                             (29123, 130515341L, 0x3257, 0x3260, 2, 35435473L, 555, 333L, 52171, 0)],self.m26_dtype)
                    
        fe_index, m26_index = correlation_functions.correlate_fm(fe_data, m26_data, self.corr_col_fm,  self.corr_row_fm, 0,1)
        
        self.assertEqual(fe_index, 3)
        self.assertEqual(m26_index, 0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCorrelation)
    unittest.TextTestRunner(verbosity=2).run(suite)


