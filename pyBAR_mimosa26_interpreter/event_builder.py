''' Class to convert M26 hit table into events using trigger timestamps.

'''
from numba import njit
import numpy as np


# important time correlation variables
FRAME_UNIT_CYCLE = 4608  # = 115.2 * 40, time for one frame in units of 40 MHz clock cylces
ROW_UNIT_CYCLE = 8  # = 115.2 * 40 / 576, time to read one row in units of 40 MHz clock cycles
LOWER_LIMIT = 48  # correct for offset between M26 40 MHz clock and 40 MHz from R/O system. Offset determined by maximizing correlation between time reference and Mimosa26.


@njit
def _correlate_ts_to_range(m26_ts_start, m26_ts_stop, ts_trigger_data, correlation_buffer, last_chunk):
    '''
    Correlate trigger timestamp to all data for which the trigger timestamp is within readout frame (integration time) of the actual
    frame data (defined by timestamp start and stop of frame data).

    Parameters:
    -----------
    m26_ts_start : array
        Start timestamp of the readout window of each frame data (row)
    m26_ts_stop : array
        Stop timestamp of the readout window of each frame data (row)
    ts_trigger_data : array
        Timestamp of the trigger word
    correlation_buffer : array
        Buffer array to store the correlated data
    last_chunk : boolean
        Indicator for last chunk
    '''

    # set all used indices to zero
    m26_ts_index = 0
    trigger_data_ts_index = 0
    buffer_index = 0

    n_m26_ts_start = m26_ts_start.shape[0]
    n_ts_trigger_data = ts_trigger_data.shape[0]

    while m26_ts_index < n_m26_ts_start and trigger_data_ts_index < n_ts_trigger_data:
        # find trigger data timestamp which lies within timestamp start and stop of M26
        if m26_ts_start[m26_ts_index] > ts_trigger_data[trigger_data_ts_index]:
            trigger_data_ts_index += 1
        elif m26_ts_stop[m26_ts_index] < ts_trigger_data[trigger_data_ts_index]:
            m26_ts_index += 1
        else:  # the case trigger data timestamo lies within start and stop M26 timestamp window
            # find point where M26 start timestamp equals trigger data timestamp
            for ts_start_index, ts_start_value in enumerate(m26_ts_start[m26_ts_index:]):
                if ts_start_value > ts_trigger_data[trigger_data_ts_index]:
                    break  # M26 timestamp start is larger than trigger timestamp

            if ts_start_index + 1 == m26_ts_start[m26_ts_index:].shape[0] and not last_chunk:
                print('WARNING: not enough M26 data in chunk!')
                break
            if correlation_buffer.shape[0] - buffer_index < ts_start_index:
                print('WARNING: chunksize for correlation buffer is too small!')
                break

            # correlate
            for index in range(ts_start_index):
                # check if trigger data timestamp is smaller than M26 timestamp stop
                if ts_trigger_data[trigger_data_ts_index] < m26_ts_stop[m26_ts_index + index]:
                    correlation_buffer[buffer_index]["m26_index"] = m26_ts_index + index
                    correlation_buffer[buffer_index]["trigger_data_index"] = trigger_data_ts_index
                    correlation_buffer[buffer_index]["m26_ts_start"] = m26_ts_start[m26_ts_index + index]
                    correlation_buffer[buffer_index]["m26_ts_stop"] = m26_ts_stop[m26_ts_index + index]
                    correlation_buffer[buffer_index]["ts_trigger_data"] = ts_trigger_data[trigger_data_ts_index]
                    buffer_index += 1
            trigger_data_ts_index += 1

    # TODO: make this nicer
    if trigger_data_ts_index < n_ts_trigger_data:
        print('WARNING: not enough trigger data in chunk!')
    if m26_ts_index < n_m26_ts_start:
        print('WARNING: not enough M26 data in chunk!')

    return correlation_buffer[:buffer_index], m26_ts_index, trigger_data_ts_index


@njit
def _correlate_to_time_ref(m26_trig_number, ref_trig_number, correlation_buffer):
    '''
    Correlate data for which M26 and time reference trigger number is the same.

    Parameters:
    -----------
    m26_trig_number : array
        Trigger numbers of Mimosa26 data
    ref_trig_number : array
        Trigger numbers of time reference data
    correlation_buffer : array
        Buffer array to store the correlated data
    '''

    m26_index = 0
    ref_index = 0
    buffer_index = 0
    n_m26_trig_number = m26_trig_number.shape[0]
    n_ref_trig_number = ref_trig_number.shape[0]

    while m26_index < n_m26_trig_number and ref_index < n_ref_trig_number:
        if m26_trig_number[m26_index] < ref_trig_number[ref_index]:
            m26_index += 1
        elif ref_trig_number[ref_index] < m26_trig_number[m26_index]:
            ref_index += 1
        else:  # The case if M26 and time reference trigger number is the same
            # get m26_index upto where M26 trigger number stays the same
            for m26_trig_number_index, m26_trig_number_value in enumerate(m26_trig_number[m26_index:]):
                if m26_trig_number[m26_index] != m26_trig_number_value:
                    break
            # get ref_index upto where time reference trigger number stays the same
            for ref_trig_number_index, ref_trig_number_value in enumerate(ref_trig_number[ref_index:]):
                if ref_trig_number[ref_index] != ref_trig_number_value:
                    break

            if m26_trig_number[m26_index] == m26_trig_number_value:
                m26_trig_number_index += 1
            if ref_trig_number[ref_index] == ref_trig_number_value:
                ref_trig_number_index += 1
            if correlation_buffer.shape[0] - buffer_index <= m26_trig_number_index:
                print('WARNING: chunksize for correlation buffer is too small!')
                break

            # correlate
            for index in range(m26_trig_number_index):
                correlation_buffer[buffer_index]["m26_data_index"] = m26_index + index  # M26 data index
                correlation_buffer[buffer_index]["time_ref_data_index"] = ref_index  # time reference data index
                correlation_buffer[buffer_index]["trg_number_time_ref"] = ref_trig_number[ref_index]  # trigger number of time reference
                buffer_index += 1

            m26_index = m26_index + m26_trig_number_index
            ref_index = ref_index + ref_trig_number_index

    if ref_index < n_ref_trig_number:
        print('WARNING: not enough time reference data in chunk!')
    if m26_index < n_m26_trig_number:
        print('WARNING: not enough M26 data in chunk!')

    return correlation_buffer[:buffer_index], m26_index, ref_index


class EventBuilder(object):
    ''' Class to convert M26 hit table into events'''

    def __init__(self, chunk_size=500000):
        self.event_table_dtype = [('event_number', '<i8'), ('event_timestamp', '<u4'), ('trigger_number', '<u4'), ('frame', "<u4"),
                                  ('m26_timestamp', '<u4'), ("column", '<u2'), ("row", '<u4'), ("tlu_y", '<u4'), ('tlu_frame', "<u4")]
        self.trigger_data_dtype = [('frame', '<u4'), ('time_stamp', '<u4'), ('trigger_number', '<u2'), ('row', '<u2')]
        self.hit_data_dtype = [('frame', '<u4'), ('time_stamp', '<u4'), ('column', '<u2'), ('row', '<u2')]
        self.correlation_buffer_dtype = [("m26_ts_start", "<i8"), ("m26_ts_stop", "<i8"), ("ts_trigger_data", "<i8"),
                                         ("m26_index", "<i8"), ("trigger_data_index", "<i8")]
        self.correlation_buffer_time_ref_dtype = [("trg_number_time_ref", "<i8"), ("m26_data_index", "<i8"), ("time_ref_data_index", "<i8")]

        # set chunksize
        self.chunk_size = chunk_size

        # reset all variables
        self.reset()

    def reset(self):
        self.event_number = 0
        self.trigger_data = np.empty(0, dtype=self.trigger_data_dtype)
        self.hit_data = np.empty(0, dtype=self.hit_data_dtype)

    def build_events_loop(self, hits, hit_data, trigger_data, plane, correlation_buffer, event_number, last_chunk):
        '''
        Loop function for event building.
        '''

        tlu_data_selection = [hits["plane"] == 255]
        trigger_data = np.append(trigger_data, hits[tlu_data_selection][["frame", "time_stamp", "trigger_number", "row"]])

        m26_data_selection = np.logical_and(hits["plane"] == plane, hits["column"] < 1152)
        hit_data = np.append(hit_data, hits[m26_data_selection][["frame", "time_stamp", "column", "row"]])

        # Calculate for each frame data (row) the actual start timestamp of this specific row readout using the frame of the readout and the row number.
        # Substraction of two frames is needed since frame data comes two frame after the beginning of the actual frame data (row) readout.
        # Additional substraction of lower limit is done in order to take into account the offset between (asynchronous) clocks of Mimosa26 and readout board.
        m26_timestamp_start = np.int64(hit_data["frame"]) * FRAME_UNIT_CYCLE + np.int64(hit_data["row"]) * ROW_UNIT_CYCLE - 2 * FRAME_UNIT_CYCLE - LOWER_LIMIT
        # Stop timestamp of actual frame data (row), one readout of specific row lasts one frame.
        m26_timestamp_stop = m26_timestamp_start + LOWER_LIMIT + FRAME_UNIT_CYCLE
        # something like timestamp of TLU word, aligned to M26 frame
        trigger_data_timestamp = np.int64(trigger_data['frame']) * FRAME_UNIT_CYCLE + np.int64(trigger_data["row"])

        # if chunksize is too small, it can happen that no TLU word is collected within data chunk.
        try:
            len(trigger_data_timestamp)
        except TypeError:
            # trigger_data_timestamp is empty
            print('WARNING: No trigger words in data found. Please choose larger chunksize!')

        # correlate trigger timestamp to M26 timestamp start and stop range
        correlation_buffer, m26_index, trigger_data_index = _correlate_ts_to_range(m26_timestamp_start,
                                                                                   m26_timestamp_stop,
                                                                                   trigger_data_timestamp,
                                                                                   correlation_buffer,
                                                                                   last_chunk)

        hit_data_out = np.empty(len(correlation_buffer), dtype=self.event_table_dtype)
        hit_data_out['event_number'] = correlation_buffer["trigger_data_index"] + event_number
        hit_data_out['event_timestamp'] = trigger_data[correlation_buffer["trigger_data_index"]]["time_stamp"]
        hit_data_out['trigger_number'] = trigger_data[correlation_buffer["trigger_data_index"]]["trigger_number"]
        hit_data_out['tlu_frame'] = trigger_data[correlation_buffer["trigger_data_index"]]["frame"]
        hit_data_out['tlu_y'] = trigger_data[correlation_buffer["trigger_data_index"]]["row"]
        hit_data_out['frame'] = hit_data[correlation_buffer["m26_index"]]["frame"]
        hit_data_out['m26_timestamp'] = hit_data[correlation_buffer["m26_index"]]["time_stamp"]
        hit_data_out['column'] = hit_data[correlation_buffer["m26_index"]]["column"]
        hit_data_out['row'] = hit_data[correlation_buffer["m26_index"]]["row"]

        return hit_data_out, hit_data, trigger_data, m26_index, trigger_data_index, event_number

    def align_with_time_ref_loop(self, m26_data, reference_data, correlation_buffer):
        '''
        Loop function for time alignment of M26 data to time reference data.
        '''

        # correlate M26 data to time refernce data
        correlation_buffer, _, _ = _correlate_to_time_ref(m26_data["trigger_number"],
                                                          reference_data["trigger_number"],
                                                          correlation_buffer)

        hit_buffer = m26_data[correlation_buffer["m26_data_index"]]

        # Take event number of time reference data
        hit_buffer['event_number'] = reference_data[correlation_buffer["time_ref_data_index"]]['event_number']

        return hit_buffer

    def build_events(self, hits, plane, last_chunk):
        '''
        Build events for M26 data. The event building is based on the assignment of one TLU data word (one trigger timestamp)
        to one M26 data frame (115.2 us). If the trigger timestamp of the TLU data word is within the range defined by
        the start and stop frame timestamp for the M26 data, it is assigned to the M26 data.
        '''
        # init correlation buffer
        correlation_buffer = np.zeros(self.chunk_size, dtype=self.correlation_buffer_dtype)
        chunk_result = self.build_events_loop(hits=hits,
                                              hit_data=self.hit_data,
                                              trigger_data=self.trigger_data,
                                              plane=plane,
                                              correlation_buffer=correlation_buffer,
                                              event_number=self.event_number,
                                              last_chunk=last_chunk)

        (hit_data_out, self.hit_data, self.trigger_data, m26_index, trigger_data_index, self.event_number) = chunk_result

        self.event_number = self.event_number + trigger_data_index
        self.trigger_data = self.trigger_data[trigger_data_index:]
        self.hit_data = self.hit_data[m26_index:]

        return hit_data_out

    def align_with_time_ref(self, m26_data, reference_data):
        '''
        Align M26 data with time reference data. In this step the event number of M26 data is corrected by using
        the event number of the time reference data based on the same trigger number. Further, only M26 data is stored
        for which an event in the time reference data exists.
        '''
        # init correlation buffer
        correlation_buffer = np.empty(self.chunk_size, dtype=self.correlation_buffer_time_ref_dtype)
        hit_data_out = self.align_with_time_ref_loop(m26_data=m26_data,
                                                     reference_data=reference_data,
                                                     correlation_buffer=correlation_buffer)

        return hit_data_out
