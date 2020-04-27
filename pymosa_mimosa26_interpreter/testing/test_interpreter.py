''' Script to check the correctness of the interpretation. Files with _orig.h5 suffix are files interpreted with Tokos
    original code. The new interpretation is checked against the old implementation of the interpreter.
'''

import os
import unittest

import tables as tb
import numpy as np

from pymosa_mimosa26_interpreter import data_interpreter
from pymosa_mimosa26_interpreter import raw_data_interpreter
# from pymosa_mimosa26_interpreter.testing.tools.test_tools import compare_h5_files

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


def create_frame_length(plane, frame_length):
    return create_m26_header(plane=plane) | (frame_length & 0x0000FFFF)


def create_row_data_word(plane, row, n_words):
    return create_m26_header(plane=plane) | (row << 4 & 0x00007FF0) | (n_words & 0x0000000F)


def create_column_data_word(plane, column, n_hits):
    return create_m26_header(plane=plane) | (column << 2 & 0x00001FFC) | (n_hits & 0x00000003)


def create_frame_trailer0(plane):
    return create_m26_header(plane=plane) | (0xaa50 & 0x0000FFFF)


def create_frame_trailer1(plane):
    return create_m26_header(plane=plane) | (((0xaa50 | plane)) & 0x0000FFFF)


class TestInterpreter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # Remove created files
        pass
        # os.remove(os.path.join(tests_data_folder, 'anemone_generated_raw_data.h5'))
        # os.remove(os.path.join(tests_data_folder, 'anemone_generated_raw_data_interpreted.h5'))
        # os.remove(os.path.join(tests_data_folder, 'anemone_interpreted.h5'))
        # os.remove(os.path.join(tests_data_folder, 'anemone_interpreted.pdf'))

    @unittest.skip("bug in create_raw_data")
    def test_interpretation(self):
        result_dtype = raw_data_interpreter.hits_dtype
        FRAME_UNIT_CYCLE = raw_data_interpreter.FRAME_UNIT_CYCLE
        ROW_UNIT_CYCLE = raw_data_interpreter.ROW_UNIT_CYCLE

        generated_raw_data_file = os.path.join(tests_data_folder, 'anemone_generated_raw_data.h5')
        generated_raw_data_interpreted_file = os.path.join(tests_data_folder, 'anemone_generated_raw_data_interpreted.h5')
        interpreted_file = os.path.join(tests_data_folder, 'anemone_interpreted.h5')

        # TODO: add multi-hits events, add events with multiple trigger, add events with out of range trigger ts
        def create_raw_data(n_events=1000, plane=0, delta_trigger_ts=8000, n_hits_per_events=1, n_events_trigger_hit=0.6, n_events_trigger_no_hit=0.3, n_events_no_trigger_hit=0.1):
            # shuffle event type: 1: event with hit and trigger; 2: event with trigger but no hit; 3: event with hit but no trigger
            event_type = np.random.choice([1, 2, 3], size=(n_events,), p=[n_events_trigger_hit, n_events_trigger_no_hit, n_events_no_trigger_hit])
            print('Generated %i events. %i events have no hit, %i events have no trigger (or no matching trigger).' % (n_events, np.sum([event_type == 2]), np.sum([event_type == 3])))
            result_array = np.zeros(shape=(np.sum([event_type == 1]),), dtype=result_dtype)
            # create random trigger time stamps
            trigger_time_stamps = np.linspace(start=14103, stop=14103 + n_events * delta_trigger_ts, num=n_events, dtype=np.int)
            hit_i = 0
            event_number = -1  # event number starts at 0
            event_status = 0

            for index in range(n_events):
                # generate row and column
                row, column = [np.random.randint(low=0, high=566), np.random.randint(low=0, high=1151)]
                # generate m26 time stamp based on event type
                row_time_stamp = np.random.randint(low=trigger_time_stamps[index] - raw_data_interpreter.FRAME_UNIT_CYCLE - raw_data_interpreter.ROW_UNIT_CYCLE, high=trigger_time_stamps[index])
                if event_type[index] != 3:
                    raw_data.append(create_tlu_word(trigger_number=index, time_stamp=trigger_time_stamps[index]))
                    event_number += 1
                if event_type[index - 1] == 3 and index != 0:
                    # if event before was event without trigger, set current event status as trigger increase error
                    event_status |= raw_data_interpreter.TRIGGER_NUMBER_ERROR
                raw_data.append(create_frame_header_low(plane=plane, m26_timestamp=row_time_stamp + 2 * FRAME_UNIT_CYCLE - ROW_UNIT_CYCLE * row + raw_data_interpreter.TIMING_OFFSET))
                raw_data.append(create_frame_header_high(plane=plane, m26_timestamp=row_time_stamp + 2 * FRAME_UNIT_CYCLE - ROW_UNIT_CYCLE * row + raw_data_interpreter.TIMING_OFFSET))
                raw_data.append(create_frame_id_low(plane=plane, m26_frame_number=index))
                raw_data.append(create_frame_id_high(plane=plane, m26_frame_number=index))
                raw_data.append(create_frame_length(plane=plane, frame_length=n_hits_per_events))  # number of data record words
                raw_data.append(create_frame_length(plane=plane, frame_length=n_hits_per_events))  # number of data record words
                if event_type[index] != 2:  # only create hit words if event with hit
                    raw_data.append(create_row_data_word(plane=plane, row=row, n_words=n_hits_per_events))
                    raw_data.append(create_column_data_word(plane=plane, column=column, n_hits=n_hits_per_events - 1))  # only one hit
                raw_data.append(create_frame_trailer0(plane=plane))
                raw_data.append(create_frame_trailer1(plane=plane))

                # write to result array
                if event_type[index] == 1:  # only write trigger hits to data file
                    result_array['plane'][hit_i] = plane
                    result_array['event_status'][hit_i] = event_status
                    result_array['event_number'][hit_i] = event_number
                    result_array['trigger_number'][hit_i] = index
                    result_array['trigger_time_stamp'][hit_i] = trigger_time_stamps[index]
                    result_array['frame_id'][hit_i] = index
                    result_array['column'][hit_i] = column
                    result_array['row'][hit_i] = row
                    result_array['row_time_stamp'][hit_i] = row_time_stamp
                    hit_i += 1
                event_status = 0

            return raw_data, result_array

        # create raw data from file
        # TODO: do raw data word creation automatically
        raw_data = []
        raw_data, result_array = create_raw_data()

        # write generated raw data to file
        filter_raw_data = tb.Filters(complib='blosc', complevel=5, fletcher32=False)
        with tb.open_file(generated_raw_data_file, 'w') as out_file_h5:
            raw_data_earray = out_file_h5.create_earray(
                where=out_file_h5.root,
                name='raw_data',
                atom=tb.UIntAtom(),
                shape=(0,), title='raw_data',
                filters=filter_raw_data)
            raw_data_earray.append(raw_data)

        # write generated interpreted file
        with tb.open_file(generated_raw_data_interpreted_file, 'w') as out_file_h5:
            hit_table = out_file_h5.create_table(
                where=out_file_h5.root,
                name='Hits',
                description=result_dtype,
                title='hit_data',
                filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hit_table.append(result_array)

        with data_interpreter.DataInterpreter(raw_data_file=generated_raw_data_file, analyzed_data_file=interpreted_file, trigger_data_format=2, analyze_m26_header_ids=[0], create_pdf=True, chunk_size=1000000) as raw_data_analysis:
            raw_data_analysis.create_occupancy_hist = True
            raw_data_analysis.create_error_hist = True
            raw_data_analysis.create_hit_table = True
            raw_data_analysis.interpret_word_table()

        # Open result and interpreter file in order to compare them. Compare only Hit fields
        with tb.open_file(generated_raw_data_interpreted_file, 'r') as in_file_h5:
            data_generated = in_file_h5.root.Hits[:]

        with tb.open_file(interpreted_file, 'r') as in_file_h5:
            data_interpreted = in_file_h5.root.Hits[:]

        # Compare with result
        for key in data_generated.dtype.names:
            if key == 'event_status':
                continue  # skip event status
            np.testing.assert_array_equal(data_generated[key], data_interpreted[key], err_msg='Column %s mismatch' % key)

        # checks_passed, error_msg = compare_h5_files(first_file=generated_raw_data_interpreted_file, second_file=interpreted_file)
        # self.assertTrue(checks_passed, msg=error_msg)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInterpreter)
    unittest.TextTestRunner(verbosity=2).run(suite)
