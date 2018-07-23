import os

from pyBAR_mimosa26_interpreter import data_interpreter


testing_path = os.path.dirname(__file__)  # Get file path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path

input_file = os.path.join(tests_data_folder, 'anemone_raw_data.h5')
time_reference_file = os.path.join(tests_data_folder, 'time_reference_interpreted_data.h5')
with data_interpreter.DataInterpreter(raw_data_file=input_file, time_reference_file=time_reference_file, trigger_data_format=2, create_pdf=True, chunk_size=1000000) as raw_data_analysis:
    raw_data_analysis.create_hit_table = True
    raw_data_analysis.interpret_word_table()
