import os
import numpy as np
import tables as tb

from pyBAR_mimosa26_interpreter import data_interpreter
from pyBAR_mimosa26_interpreter import raw_data_interpreter

from hypothesis import given, settings
import hypothesis.strategies as st
import unittest


testing_path = os.path.dirname(__file__)  # Get file path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/'))  # Set test data path


def create_tlu_word(trigger_number, time_stamp):
    return ((time_stamp << 16) & (0x7FFF0000)) | (trigger_number & 0x0000FFFF) | (1 << 31 & 0x80000000)

def create_m26_header(plane, data_loss=False):
    return (0x20 << 24 & 0xFF000000) | (plane << 20 & 0x00F00000)

def create_frame_header_low(plane, m26_timestamp):
    return create_m26_header(plane=plane) | (m26_timestamp & 0x0000FFFF) | (1 << 16 & 0x00010000)

def create_frame_header_high(plane, m26_timestamp):
    return create_m26_header(plane=plane) | (((m26_timestamp & 0xFFFF0000) >> 16) & 0x0000FFFF)

def create_frame_id_low(plane, m26_frame_number):
    return create_m26_header(plane=plane) | (m26_frame_number & 0x0000FFFF)

def create_frame_id_high(plane, m26_frame_number):
    return create_m26_header(plane=plane) | (((m26_frame_number & 0xFFFF0000) >> 16) & 0x0000FFFF)


result_dtype = raw_data_interpreter.hits_dtype

# create test data
test_data_length = 6
result_array = np.zeros(shape=(test_data_length,), dtype=result_dtype)
result_array['plane'] = np.array([1] * 6)
result_array['event_number'] = np.array([1, 2, 3, 4, 4, 5])
result_array['trigger_number'] = np.array([1, 2, 3, 4, 4, 5])
result_array['trigger_time_stamp'] = np.array([6731, 4800, 23000, 16000, 16000, 5800])
result_array['column'] = np.array([45, 256, 407, 3, 1056, 50])
result_array['row'] = np.array([501, 23, 68, 385, 185, 77])

# additional information which is needed
frame_number = np.array([1, 2, 2, 3, 3, 7])
m26_timestamp = np.array([6890, 41450, 76467, 102567, 137346, 172500])

# create result file
with tb.open_file('anemone_raw_data_interpreted_result.h5', 'w') as out_file_h5:
    hit_table = out_file_h5.create_table(where=out_file_h5.root,
                                         name='Hits',
                                         description=result_dtype,
                                         title='hit_data',
                                         filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
    hit_table.append(result_array)

# create raw data from file
raw_data = []
for index, data in enumerate(result_array):
    raw_data += create_tlu_word(trigger_number=data['trigger_number'], time_stamp=data['trigger_time_stamp'])
    raw_data += create_frame_header_low(plane=data['plane'], m26_timestamp=m26_timestamp[index])
    raw_data += create_frame_header_high(plane=data['plane'], m26_timestamp=m26_timestamp[index])


input_file = os.path.join(tests_data_folder, 'anemone_raw_data.h5')
time_reference_file = os.path.join(tests_data_folder, 'time_reference_interpreted_data.h5')
with data_interpreter.DataInterpreter(raw_data_file=input_file, time_reference_file=time_reference_file, trigger_data_format=2, create_pdf=True, chunk_size=1000000) as raw_data_analysis:
    raw_data_analysis.create_hit_table = True
    raw_data_analysis.interpret_word_table()
