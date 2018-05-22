''' Script to check the correctness of the interpretation. Files with _orig.h5 suffix are files interpreted with Tokos
    original code. The new interpretation is checked against the old implementation of the interpreter.
'''

import os
import unittest
import tables as tb
import numpy as np

from pyBAR_mimosa26_interpreter import data_interpreter

testing_path = os.path.dirname(__file__)  # Get file path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path


class TestInterpretation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(self):  # Remove created files
        os.remove(tests_data_folder + r'/anemone_raw_data_interpreted.h5')

    def test_interpretation(self):
        input_file = tests_data_folder + r'/anemone_raw_data.h5'
        time_reference_file = tests_data_folder + r'/time_reference_interpreted_data.h5'
        with data_interpreter.DataInterpreter(raw_data_file=input_file, time_reference_file=time_reference_file, trigger_data_format=2, create_pdf=False) as raw_data_analysis:
            raw_data_analysis.create_hit_table = True
            raw_data_analysis.interpret_word_table()
            # raw_data_analysis.interprete_hit_table()

        with tb.open_file(tests_data_folder + r'/anemone_raw_data_interpreted_orig.h5', 'r') as in_file_h5_orig:
            data_orig = in_file_h5_orig.root.Hits[:]

        with tb.open_file(tests_data_folder + r'/anemone_raw_data_interpreted.h5', 'r') as in_file_h5:
            data = in_file_h5.root.Hits[:]

        # Hits are sorted differently per plane and field names are completely different, thus loop and selection is needed
        for plane in range(1, 7):
            ts_orig = data_orig['timestamp'][data_orig['plane'] == plane]
            frame_orig = data_orig['mframe'][data_orig['plane'] == plane]
            trg_number_orig = data_orig['tlu'][data_orig['plane'] == plane]
            column_orig = data_orig['x'][data_orig['plane'] == plane]
            row_orig = data_orig['y'][data_orig['plane'] == plane]
            ts = data['time_stamp'][data['plane'] == plane]
            frame = data['frame'][data['plane'] == plane]
            trg_number = data['trigger_number'][data['plane'] == plane]
            column = data['column'][data['plane'] == plane]
            row = data['row'][data['plane'] == plane]
            np.testing.assert_array_equal(ts_orig, ts, err_msg='Timestamp array mismatch for plane: %d' % (plane))
            np.testing.assert_array_equal(frame_orig, frame, err_msg='Frame array mismatch for plane: %d' % (plane))
            np.testing.assert_array_equal(trg_number_orig, trg_number, err_msg='Trigger Number array mismatch for plane: %d' % (plane))
            np.testing.assert_array_equal(column_orig, column, err_msg='Column array mismatch for plane: %d' % (plane))
            np.testing.assert_array_equal(row_orig, row, err_msg='Row array mismatch for plane: %d' % (plane))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInterpretation)
    unittest.TextTestRunner(verbosity=2).run(suite)
