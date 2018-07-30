import os

from pyBAR_mimosa26_interpreter import data_interpreter
from pyBAR_mimosa26_interpreter.testing.tools.test_tools import compare_h5_files


testing_path = os.path.dirname(__file__)  # Get file path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path

input_file = os.path.join(tests_data_folder, 'anemone_raw_data.h5')
reference_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted_orig.h5')
for size in range(1000, 1020):
    temp_output_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted_chunk_size_%d.h5' % size)
    with data_interpreter.DataInterpreter(raw_data_file=input_file, analyzed_data_file=temp_output_file, trigger_data_format=2, create_pdf=False, chunk_size=size) as raw_data_analysis:
        raw_data_analysis.create_hit_table = True
        raw_data_analysis.interpret_word_table()

    checks_passed, error_msg = compare_h5_files(reference_file, temp_output_file, node_names=None, detailed_comparison=True, exact=True, rtol=1e-5, atol=1e-8, chunk_size=1000000)
    if checks_passed:
        print "Chunk size %d" % size, "OK"
    else:
        print "Chunk size %d" % size, "FAILED:", error_msg

    os.remove(temp_output_file)
