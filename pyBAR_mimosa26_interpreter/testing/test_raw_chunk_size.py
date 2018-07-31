import os

import numpy as np
import tables as tb

from pyBAR_mimosa26_interpreter import data_interpreter
from pyBAR_mimosa26_interpreter.testing.tools.test_tools import compare_h5_files

chunk_size = 999999
testing_path = os.path.dirname(__file__)  # Get file path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path

input_file = os.path.join(tests_data_folder, 'anemone_raw_data.h5')
reference_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted_orig.h5')
start_chunk_size = 1000
iterations = 50
checks_passed = True
for curr_chunk_size in range(start_chunk_size, start_chunk_size + iterations):
    if not checks_passed:
        break
    temp_output_file = os.path.join(tests_data_folder, 'anemone_raw_data_interpreted_chunk_size_%d.h5' % curr_chunk_size)
    with data_interpreter.DataInterpreter(raw_data_file=input_file, analyzed_data_file=temp_output_file, trigger_data_format=2, create_pdf=False, chunk_size=curr_chunk_size) as raw_data_analysis:
        raw_data_analysis.create_hit_table = True
        raw_data_analysis.interpret_word_table()

    checks_passed, error_msg = compare_h5_files(reference_file, temp_output_file, node_names=None, detailed_comparison=True, exact=True, rtol=1e-5, atol=1e-8, chunk_size=1000000)
    if checks_passed:
        print "Chunk size %d" % curr_chunk_size, "OK"
    else:
        print "Chunk size %d" % curr_chunk_size, "FAILED:", error_msg

    with tb.open_file(temp_output_file, 'r') as in_file_h5:
        last_event_number = np.zeros(shape=1, dtype=np.int64)
        for i in range(0, in_file_h5.root.Hits.shape[0], chunk_size):
            hits_chunk = in_file_h5.root.Hits.read(i, i + chunk_size)
            if np.any(np.diff(np.concatenate((last_event_number, hits_chunk['event_number']))) < 0):
                print 'Event number not increasing!'
            last_event_number = hits_chunk['event_number'][-1:]

    os.remove(temp_output_file)
