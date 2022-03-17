import os
import unittest

from pymosa_mimosa26_interpreter import data_interpreter
from pymosa_mimosa26_interpreter.testing.tools.test_tools import compare_h5_files


class TestRawChunkSize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_output_files = []

    @classmethod
    def tearDownClass(cls):  # Remove created files
        for temp_output_file in cls.temp_output_files:
            os.remove(temp_output_file)

    def test_varying_chunk_sizes(self):
        testing_path = os.path.dirname(__file__)  # Get file path
        tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path
        input_file = os.path.join(tests_data_folder, 'anemone_raw_data.h5')
        reference_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted.h5')
        for curr_chunk_size in range(1000, 2000, 101):
            temp_output_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted_chunk_size_%d.h5' % curr_chunk_size)
            self.temp_output_files.append(temp_output_file)
            with data_interpreter.DataInterpreter(raw_data_file=input_file, analyzed_data_file=temp_output_file, trigger_data_format=2, create_pdf=False, chunk_size=curr_chunk_size) as interpreter:
                interpreter.create_hit_table = True
                interpreter.create_occupancy_hist = True
                interpreter.create_error_hist = True
                interpreter.interpret_word_table()

            checks_passed, error_msg = compare_h5_files(reference_file, temp_output_file, node_names=None, detailed_comparison=True, exact=True, rtol=1e-5, atol=1e-8, chunk_size=1000000)
            self.assertTrue(checks_passed, msg=error_msg)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRawChunkSize)
    unittest.TextTestRunner(verbosity=2).run(suite)
