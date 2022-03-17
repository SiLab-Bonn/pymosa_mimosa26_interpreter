''' Class to convert Mimosa26 raw data.

General structure of Mimosa26 raw data (32 bit words):
 - First 8 bits are always 0x20 (Mimosa26 HEADER)
 - Next 4 bits are plane number 1 - 6 (plane identifier)
 - Next 2 bits always zero
 - Next two bits contain: data loss flag and frame start flag
 - Rest of 16 bits are actual data words

The raw data structure of Mimosa26 data looks as follows:
 - Frame header HIGH and LOW (contains timestamp, generated from R/O) [word index 0 + 1]
 - Frame number HIGH and LOW (frame number of Mimosa26) [word index 2 + 3]
 - Frame length HIGH and LOW (number of Mimosa26) [word index 4 + 5]
 - Hit data (column and row of hit pixel)
 - ...
 - ...
 - Frame trailer HIGH and LOW (indicates end of Mimosa26 frame) [word index 6 + 7]

'''
import numba
from numba import njit
import numpy as np


MIMOSA_FRAME_CYCLE = 115.2  # us
MIMOSA_FREQ = 40  # MHz
N_ROWS_MIMOSA = 576  # Number of rows
FRAME_UNIT_CYCLE = int(MIMOSA_FRAME_CYCLE * MIMOSA_FREQ)  # = 4608, time for one frame in units of 40 MHz clock cylces
ROW_UNIT_CYCLE = int(MIMOSA_FRAME_CYCLE * MIMOSA_FREQ / N_ROWS_MIMOSA)  # = 8, time to read one row in units of 40 MHz clock cycles
TIMING_OFFSET = -112  # Correct for offset between M26 40 MHz clock and 40 MHz from R/O system. Offset determined by maximum correlation between the time reference and Mimosa26 telescope.
MAX_BUFFER_TIME_SLIP = 5  # max. time (in seconds) for storing hits in buffer before they get removed if no trigger appears
DEFAULT_PYMOSA_M26_HEADER_IDS = [1, 2, 3, 4, 5, 6]  # Default header IDs for the Mimosa26 data generated by the pymosa software. The header IDs are set in the pymosa readout software.

hits_dtype = np.dtype([
    ('plane', '<u1'),
    ('event_number', '<i8'),
    ('trigger_number', '<i8'),
    ('trigger_time_stamp', '<i8'),
    ('row_time_stamp', '<i8'),
    ('frame_id', '<i8'),
    ('column', '<u2'),
    ('row', '<u2'),
    ('event_status', '<u4')])

telescope_data_dtype = np.dtype([
    ('plane', '<u1'),
    ('time_stamp', '<i8'),
    ('frame_id', '<i8'),
    ('column', '<u2'),
    ('row', '<u2'),
    ('frame_status', '<u4')])

trigger_data_dtype = np.dtype([
    ('event_number', '<i8'),
    ('trigger_number', '<i8'),
    ('trigger_time_stamp', '<i8'),
    ('trigger_status', '<u4')])

# Error codes
TRIGGER_NUMBER_ERROR = 0x00000001  # Trigger number has not increased by one
NO_TRIGGER_WORD_ERROR = 0x00000002  # Event has no trigger word associated
TRIGGER_TIMESTAMP_OVERFLOW = 0x00000004  # Indicating the overflow of the trigger timestamp
TRIGGER_NUMBER_OVERFLOW = 0x00000008  # Indicating the overflow of the trigger number
DATA_ERROR = 0x00000010  # Indicating any occurrence of data errors in the Momosa26 protocol (e.g., invalid column/row, invalid data length, data loss)
TIMESTAMP_OVERFLOW = 0x00000020  # Indicating the overflow of the Mimosa26 timestamp
FRAME_ID_OVERFLOW = 0x00000040  # Indicating the overflow of the Mimosa26 frame ID
OVERFLOW_FLAG = 0x00000080  # Indicating the occurrence of the overflow flag for a particular Mimosa26 row


# Mimosa26 raw data
@njit
def is_mimosa_data(word):  # Check for Mimosa data word
    return (0xff000000 & word) == 0x20000000


@njit
def get_plane_number(word):  # There are 6 planes in the stream, starting from 1; return plane number
    return (word >> 20) & 0xf


# Frame header
@njit
def is_frame_header(word):  # Check if frame header high word (frame start flag is set by R/0)
    return (0x00010000 & word) == 0x00010000


@njit
def is_data_loss(word):  # Indicates data loss
    return (0x00020000 & word) == 0x00020000


@njit
def get_m26_timestamp_low(word):  # Timestamp of Mimosa26 data from frame header low (generated by R/0)
    return 0x0000ffff & word


@njit
def get_m26_timestamp_high(word):  # Timestamp of Mimosa26 data from frame header high (generated by R/0)
    return (0x0000ffff & word) << 16


@njit
def is_frame_header0(word):  # Check if frame header0 word
    return (0x0000ffff & word) == 0x00005555


@njit
def is_frame_header1(word, plane):  # Check if frame header1 word for the actual plane
    return (0x0000ffff & word) == (0x00005550 | plane)


# Frame counter
@njit
def get_frame_id_low(word):  # Get the frame id from the frame id low word
    return 0x0000ffff & word


@njit
def get_frame_id_high(word):  # Get the frame id from the frame id high word
    return (0x0000ffff & word) << 16


# Data length
@njit
def get_frame_length(word):  # Get length of Mimosa26 frame
    return (0x0000ffff & word)


# Status / line word
@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return 0x0000000f & word


@njit
def get_row(word):  # Extract row from Mimosa26 hit word
    return (0x00007ff0 & word) >> 4


@njit
def has_overflow(word):
    return (0x00008000 & word) != 0


# State word
@njit
def get_n_hits(word):  # Returns the number of hits given by actual column word
    return 0x00000003 & word


@njit
def get_column(word):  # Extract column from Mimosa26 hit word
    return (0x00001ffc & word) >> 2


# Frame trailer
@njit
def is_frame_trailer0(word):  # Check if frame trailer0 word
    return (0x0000ffff & word) == 0xaa50


@njit
def is_frame_trailer1(word, plane):  # Check if frame trailer1 word for the actual plane
    return (0x0000ffff & word) == (0xaa50 | plane)


# Trigger words
@njit
def is_trigger_word(word):  # Check if TLU word (trigger)
    return (0x80000000 & word) == 0x80000000


@njit
def get_trigger_timestamp(word):  # Get timestamp of TLU word
    return (word & 0x7fff0000) >> 16


@njit
def get_trigger_number(word, trigger_data_format):  # Get trigger number of TLU word
    if trigger_data_format == 2:
        return word & 0x0000ffff
    else:
        return word & 0x7fffffff


class RawDataInterpreter(object):
    ''' Class to convert the raw data chunks to hits'''

    def __init__(self, analyze_m26_header_ids=None):
        '''
        Parameters:
        -----------
        analyze_m26_header_ids : list
            List of Mimosa26 header IDs that will be interpreted.
            If None, the value defaults to the global value raw_data_interpreter.DEFAULT_PYMOSA_M26_HEADER_IDS.
        '''
        if analyze_m26_header_ids is None:
            self.analyze_m26_header_ids = DEFAULT_PYMOSA_M26_HEADER_IDS
        else:
            self.analyze_m26_header_ids = analyze_m26_header_ids
        for analyze_m26_header_id in self.analyze_m26_header_ids:
            if analyze_m26_header_id < 0 or analyze_m26_header_id >= 2**16:
                raise ValueError('Invalid header ID.')
        self.analyze_m26_header_ids = np.asarray(self.analyze_m26_header_ids, dtype=np.uint16)
        self.plane_id_to_index = -1 * np.ones(shape=max(self.analyze_m26_header_ids) + 1, dtype=np.int32)
        for plane_index, plane_id in enumerate(self.analyze_m26_header_ids):
            self.plane_id_to_index[plane_id] = plane_index
        self.reset()

    def reset(self):  # Reset variables
        # Temporary arrays
        self.trigger_data = np.zeros(shape=0, dtype=trigger_data_dtype)
        self.trigger_data_index = np.int64(-1)
        self.telescope_data = np.zeros(shape=0, dtype=telescope_data_dtype)
        self.telescope_data_index = np.int64(-1)

        # Raw data interpreter
        # Per frame variables
        self.m26_frame_ids = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.int64)  # The Mimosa26 frame ID of the actual frame
        self.m26_frame_length = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.uint32)  # The number of "useful" data words for the actual frame
        self.m26_data_loss = np.ones(len(self.analyze_m26_header_ids), dtype=np.bool_)  # The data loss status for the actual frame
        self.m26_word_index = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.uint32)  # The word index per device of the actual frame
        self.m26_timestamps = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.int64)  # The timestamp for each plane (in units of 40 MHz)
        self.last_m26_timestamps = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.int64)
        self.m26_n_words = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.uint32)  # The number of words containing column / row info
        self.m26_rows = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.uint32)  # The actual readout row (rolling shutter)
        self.m26_frame_status = np.zeros(shape=len(self.analyze_m26_header_ids), dtype=np.uint32)  # The status flags for the actual frames
        self.last_completed_m26_frame_ids = -1 * np.ones(shape=len(self.analyze_m26_header_ids), dtype=np.int64)  # The status if the frame is complete for the actual frame
        # Per event variables
        self.event_number = np.int64(-1)  # The event number of the actual trigger, event number starts at 0
        self.trigger_number = np.int64(-1)  # The trigger number of the actual trigger
        self.trigger_timestamp = np.int64(0)  # The trigger timestamp of the actual trigger

        # Event builder
        self.hits = np.zeros(shape=0, dtype=hits_dtype)
        self.hits_index = np.int64(-1)

        # Properties
        self._add_missing_events = False
        self._timing_offset = TIMING_OFFSET

    @property
    def add_missing_events(self):
        return self._add_missing_events

    @add_missing_events.setter
    def add_missing_events(self, value):
        self._add_missing_events = bool(value)

    @property
    def timing_offset(self):
        return self._timing_offset

    @timing_offset.setter
    def timing_offset(self, value):
        self._timing_offset = int(value)

    def interpret_raw_data(self, raw_data=None, build_all_events=False):
        ''' Converting the raw data array to a hit array.
        The is the only function that needs to be called to convert the raw data.

        Parameters:
        -----------
        raw_data : np.array
            The array with the raw data words.
        build_all_events : bool
            If True, build all events from the remaining trigger_data and telescope_data_array.
            Use this only after the last raw data chunk to receive the the remaining events in the buffers.
        '''
        if raw_data is None:
            raw_data = np.zeros(shape=0, dtype=np.uint32)
        if self.telescope_data_index != -1:
            telescope_data_index_start = self.telescope_data_index + 1
        else:
            telescope_data_index_start = 0
        # Analyze raw data
        self.trigger_data, self.trigger_data_index, self.telescope_data, self.telescope_data_index, self.m26_frame_ids, self.m26_frame_length, self.m26_data_loss, self.m26_word_index, self.m26_timestamps, self.last_m26_timestamps, self.m26_n_words, self.m26_rows, self.m26_frame_status, self.last_completed_m26_frame_ids, self.event_number, self.trigger_number, self.trigger_timestamp = _interpret_raw_data(
            raw_data=raw_data,
            trigger_data=self.trigger_data,
            trigger_data_index=self.trigger_data_index,
            telescope_data=self.telescope_data,
            telescope_data_index=self.telescope_data_index,
            m26_frame_ids=self.m26_frame_ids,
            m26_frame_length=self.m26_frame_length,
            m26_data_loss=self.m26_data_loss,
            m26_word_index=self.m26_word_index,
            m26_timestamps=self.m26_timestamps,
            last_m26_timestamps=self.last_m26_timestamps,
            m26_n_words=self.m26_n_words,
            m26_rows=self.m26_rows,
            m26_frame_status=self.m26_frame_status,
            last_completed_m26_frame_ids=self.last_completed_m26_frame_ids,
            event_number=self.event_number,
            trigger_number=self.trigger_number,
            trigger_timestamp=self.trigger_timestamp,
            add_missing_events=self.add_missing_events,
            build_all_events=build_all_events,
            analyze_m26_header_ids=self.analyze_m26_header_ids,
            plane_id_to_index=self.plane_id_to_index)

        # Get data from telescope (just hit data, no assignment to events or data multiplication)
        telescope_data = self.telescope_data[telescope_data_index_start:self.telescope_data_index + 1].copy()

        # Build events
        self.trigger_data, self.trigger_data_index, self.telescope_data, self.telescope_data_index, self.hits, self.hits_index = _build_events(
            trigger_data=self.trigger_data,
            trigger_data_index=self.trigger_data_index,
            telescope_data=self.telescope_data,
            telescope_data_index=self.telescope_data_index,
            hits=self.hits,
            hits_index=self.hits_index,
            last_completed_m26_frame_ids=self.last_completed_m26_frame_ids,
            timing_offset=self.timing_offset,
            build_all_events=build_all_events,
            analyze_m26_header_ids=self.analyze_m26_header_ids,
            plane_id_to_index=self.plane_id_to_index)
        # Create a copy of the hits array that is returned
        hits = self.hits[:self.hits_index + 1].copy()
        self.hits_index -= (self.hits_index + 1)

        return hits, telescope_data


@njit(locals={'trigger_data_index': numba.int64, 'telescope_data_index': numba.int64, 'trigger_status': numba.uint32, 'last_trigger_number': numba.int64, 'last_trigger_timestamp': numba.int64, 'n_missing_events': numba.uint32})
def _interpret_raw_data(raw_data, trigger_data, trigger_data_index, telescope_data, telescope_data_index, m26_frame_ids, m26_frame_length, m26_data_loss, m26_word_index, m26_timestamps, last_m26_timestamps, m26_n_words, m26_rows, m26_frame_status, last_completed_m26_frame_ids, event_number, trigger_number, trigger_timestamp, add_missing_events, build_all_events, analyze_m26_header_ids, plane_id_to_index):
    ''' This function is interpreting the Mimosa26 telescope raw data and creates temporary trigger and telescope data arrays.
    The interpreter checks for trigger and Mimosa26 data errors.

    Parameters:
    -----------
    raw_data : np.array
        The array with the raw data words.
    TBD
    '''
    # Loop over the raw data words
    for raw_data_word in raw_data:
        if is_mimosa_data(raw_data_word):  # Check if word is from Mimosa26.
            # Check to which plane the data belongs
            plane_id = get_plane_number(raw_data_word)  # The actual_plane if the actual word belongs to (0 to 5)
            for analyze_m26_header_id in analyze_m26_header_ids:
                if plane_id == analyze_m26_header_id:
                    break
            else:
                continue  # Do not interpret data of planes which should be skipped
            plane_index = plane_id_to_index[plane_id]
            # In the following, interpretation of the raw data words of the actual plane
            # Check for data loss bit set by the M26 RX FSM
            if is_data_loss(raw_data_word):
                # Setting the data loss flag to true.
                # The data loss bit is set by the M26 RX FSM.
                # The bit is set only once after each data loss, i.e.,
                # the first data word after the lost data words.
                m26_data_loss[plane_index] = True
            if is_frame_header(raw_data_word):  # New frame for actual plane, M26 timestamp (LSB), frame header0
                # Get Mimosa26 timestamp from raw data word (LSB)
                last_m26_timestamps[plane_index] = m26_timestamps[plane_index]
                m26_timestamps[plane_index] = (m26_timestamps[plane_index] & 0x7fffffffffff0000) | get_m26_timestamp_low(raw_data_word)
                m26_word_index[plane_index] = 0
                # Reset parameters after header
                m26_frame_length[plane_index] = 0
                m26_n_words[plane_index] = 0
                # Set the status bits for priviously incomplete frames
                index = telescope_data_index
                while index >= 0:
                    if telescope_data[index]['plane'] == plane_id:
                        if telescope_data[index]['frame_id'] > last_completed_m26_frame_ids[plane_index]:
                            telescope_data[index]['frame_status'] |= DATA_ERROR
                        else:
                            break
                    index -= 1
                m26_data_loss[plane_index] = False
                m26_frame_status[plane_index] = 0
            elif m26_data_loss[plane_index] is True:  # Trash data
                # Nothing to do, do not trust data
                continue
            else:  # Interpreting M26 raw data
                m26_word_index[plane_index] += 1
                if m26_word_index[plane_index] == 1:  # Mimosa26 timestamp, M26 timestamp (MSB), frame header1
                    # Check for 32bit timestamp overflow
                    if m26_timestamps[plane_index] >= 0 and get_m26_timestamp_high(raw_data_word) < (m26_timestamps[plane_index] & 0x00000000ffff0000):
                        m26_frame_status[plane_index] |= TIMESTAMP_OVERFLOW
                        m26_timestamps[plane_index] = np.int64(2**32) + m26_timestamps[plane_index]
                    # Get Mimosa26 timestamp from raw data word (MSB)
                    m26_timestamps[plane_index] = get_m26_timestamp_high(raw_data_word) | (m26_timestamps[plane_index] & 0x7fffffff0000ffff)
                elif m26_word_index[plane_index] == 2:  # Mimosa26 frame ID
                    # Get Mimosa26 frame ID from raw data word (LSB)
                    m26_frame_ids[plane_index] = (m26_frame_ids[plane_index] & 0x7fffffffffff0000) | get_frame_id_low(raw_data_word)
                elif m26_word_index[plane_index] == 3:  # Mimosa26 frame ID
                    # Check for 32bit frame ID overflow
                    if m26_frame_ids[plane_index] >= 0 and get_frame_id_high(raw_data_word) < (m26_frame_ids[plane_index] & 0x00000000ffff0000):
                        m26_frame_status[plane_index] |= FRAME_ID_OVERFLOW
                        m26_frame_ids[plane_index] = np.int64(2**32) + m26_frame_ids[plane_index]
                    # Get Mimosa26 frame ID from raw data word (MSB)
                    m26_frame_ids[plane_index] = get_frame_id_high(raw_data_word) | (m26_frame_ids[plane_index] & 0x7fffffff0000ffff)
                elif m26_word_index[plane_index] == 4:  # Mimosa26 frame length
                    m26_frame_length[plane_index] = get_frame_length(raw_data_word)
                    if m26_frame_length[plane_index] > 570:  # Defined in the Mimosa26 protocol, no more than 570 "useful" data words
                        m26_data_loss[plane_index] = True
                        continue
                elif m26_word_index[plane_index] == 5:  # Mimosa26 frame length, a second time
                    if m26_frame_length[plane_index] != get_frame_length(raw_data_word):  # DO0 & DO1 should always have the same data length
                        m26_data_loss[plane_index] = True
                        continue
                    else:
                        m26_frame_length[plane_index] += get_frame_length(raw_data_word)
                elif m26_word_index[plane_index] == 5 + m26_frame_length[plane_index] + 1:  # Frame trailer0
                    if not is_frame_trailer0(raw_data_word):
                        m26_data_loss[plane_index] = True
                        continue
                elif m26_word_index[plane_index] == 5 + m26_frame_length[plane_index] + 2:  # Frame trailer1
                    if not is_frame_trailer1(raw_data_word, plane=plane_id):
                        m26_data_loss[plane_index] = True
                        continue
                    else:
                        last_completed_m26_frame_ids[plane_index] = m26_frame_ids[plane_index]
                elif m26_word_index[plane_index] > 5 + m26_frame_length[plane_index] + 2:  # Ignore any occurrence of additional raw data words
                    m26_data_loss[plane_index] = True
                    continue
                else:  # Column / Row words (actual data word with hits)
                    if m26_n_words[plane_index] == 0:  # First word contains the row info and the number of data words for this row
                        if m26_word_index[plane_index] == 5 + m26_frame_length[plane_index]:  # Always even amount of words or this fill word is used
                            # Ignore this fill word
                            continue
                        else:
                            m26_n_words[plane_index] = get_n_words(raw_data_word)
                            m26_rows[plane_index] = get_row(raw_data_word)  # Get row from data word
                            if m26_rows[plane_index] >= 576:  # Row overflow
                                m26_data_loss[plane_index] = True
                                continue
                        if has_overflow(raw_data_word):
                            m26_frame_status[plane_index] |= OVERFLOW_FLAG  # set overflow bit
                        else:
                            m26_frame_status[plane_index] & ~OVERFLOW_FLAG  # unset overflow bit
                    else:
                        m26_n_words[plane_index] = m26_n_words[plane_index] - 1  # Count down the words
                        n_hits = get_n_hits(raw_data_word)
                        column = get_column(raw_data_word)  # Get column from data word
                        if column >= 1152:  # Column overflow
                            m26_data_loss[plane_index] = True
                            continue
                        for k in range(n_hits + 1):
                            if column + k >= 1152:
                                m26_data_loss[plane_index] = True
                                break
                            # Increase index
                            telescope_data_index += 1
                            # extend telescope data array if neccessary
                            if telescope_data_index >= telescope_data.shape[0]:
                                # remove old hit data from array for each plane individually. Prevents the case that telescope data array gets too big in case
                                # time until next trigger is very large, since telescope data has to be buffered until next trigger.
                                select = (telescope_data['plane'] == plane_id)
                                select &= (telescope_data['time_stamp'] < (m26_timestamps[plane_index] - MAX_BUFFER_TIME_SLIP * MIMOSA_FREQ * 10**6))
                                count_outdated = np.sum(select)
                                if count_outdated:
                                    telescope_data = telescope_data[~select]
                                    telescope_data_index = telescope_data_index - count_outdated
                                # extend telescope data array if neccessary
                                telescope_data_tmp = np.zeros(shape=max(1, int(raw_data.shape[0] / 2)), dtype=telescope_data_dtype)
                                telescope_data = np.concatenate((telescope_data, telescope_data_tmp))

                            # Store hits
                            telescope_data[telescope_data_index]['plane'] = plane_id
                            telescope_data[telescope_data_index]['time_stamp'] = m26_timestamps[plane_index]
                            telescope_data[telescope_data_index]['frame_id'] = m26_frame_ids[plane_index]
                            telescope_data[telescope_data_index]['column'] = column + k
                            telescope_data[telescope_data_index]['row'] = m26_rows[plane_index]
                            telescope_data[telescope_data_index]['frame_status'] = m26_frame_status[plane_index]
        elif is_trigger_word(raw_data_word):  # Raw data word is TLU/trigger word
            # Reset trigger status
            trigger_status = 0
            # Get latest telescope timestamp and set trigger timestamp
            last_trigger_timestamp = trigger_timestamp
            # Get largest M26 timestamp
            for tmp_plane_index, _ in enumerate(analyze_m26_header_ids):
                if last_m26_timestamps[tmp_plane_index] > trigger_timestamp:
                    trigger_timestamp = last_m26_timestamps[tmp_plane_index]
            # Calculating 63bit timestamp from 15bit trigger timestamp
            # and last telescope timestamp (frame header timestamp).
            # Assumption: the telescope timestamp is updated more frequent than
            # the 15bit trigger timestamp can overflow. The frame is occurring
            # every 4608 clock cycles (115.2 us).
            # Get trigger timestamp from raw data word
            trigger_timestamp = (0x7fffffffffff8000 & trigger_timestamp) | get_trigger_timestamp(raw_data_word)
            # Check for 15bit trigger timestamp overflow
            if last_trigger_timestamp >= 0 and trigger_timestamp <= last_trigger_timestamp:
                trigger_status |= TRIGGER_TIMESTAMP_OVERFLOW
                trigger_timestamp = np.int64(2**15) + trigger_timestamp
            # Copy of trigger number
            last_trigger_number = trigger_number
            # Check for 16bit trigger number overflow
            if trigger_number >= 0 and get_trigger_number(raw_data_word, trigger_data_format=2) <= (trigger_number & 0x000000000000ffff):
                trigger_status |= TRIGGER_NUMBER_OVERFLOW
                trigger_number = np.int64(2**16) + trigger_number
            # Get trigger number from raw data word
            if trigger_number < 0:
                trigger_number = get_trigger_number(raw_data_word, trigger_data_format=2)
            else:
                trigger_number = (0x7fffffffffff0000 & trigger_number) | get_trigger_number(raw_data_word, trigger_data_format=2)
            # Check validity of trigger number
            # Trigger number has to increase by 1
            if trigger_data_index >= 0:
                # Check if trigger number has increased by 1
                if last_trigger_number < 0:
                    n_missing_events = 0
                else:
                    n_missing_events = trigger_number - (last_trigger_number + 1)
                if n_missing_events != 0:
                    if n_missing_events > 0 and add_missing_events:
                        for i in range(n_missing_events):
                            # Increase index
                            trigger_data_index += 1
                            # extend trigger data array if neccessary
                            if trigger_data_index >= trigger_data.shape[0]:
                                trigger_data_tmp = np.zeros(shape=max(1, int(raw_data.shape[0] / 6)), dtype=trigger_data_dtype)
                                trigger_data = np.concatenate((trigger_data, trigger_data_tmp))
                            # Increase event number
                            event_number += 1
                            # Store trigger data
                            trigger_data[trigger_data_index]['event_number'] = event_number  # Timestamp of TLU word
                            trigger_data[trigger_data_index]['trigger_time_stamp'] = -1  # Timestamp of TLU word
                            trigger_data[trigger_data_index]['trigger_number'] = trigger_data[trigger_data_index - 1]['trigger_number'] + 1 + i
                            trigger_data[trigger_data_index]['trigger_status'] = NO_TRIGGER_WORD_ERROR  # Trigger status
                    else:
                        trigger_status |= TRIGGER_NUMBER_ERROR
            # Increase index
            trigger_data_index += 1
            # extend trigger data array if neccessary
            if trigger_data_index >= trigger_data.shape[0]:
                trigger_data_tmp = np.zeros(shape=max(1, int(raw_data.shape[0] / 6)), dtype=trigger_data_dtype)
                trigger_data = np.concatenate((trigger_data, trigger_data_tmp))
            # Increase event number
            event_number += 1
            # Store trigger data
            trigger_data[trigger_data_index]['event_number'] = event_number  # Timestamp of TLU word
            trigger_data[trigger_data_index]['trigger_number'] = trigger_number
            trigger_data[trigger_data_index]['trigger_time_stamp'] = trigger_timestamp  # Timestamp of TLU word
            trigger_data[trigger_data_index]['trigger_status'] = trigger_status  # Trigger status
        else:  # Raw data contains unknown word, neither M26 nor TLU word
            for tmp_plane_index, _ in enumerate(analyze_m26_header_ids):
                m26_data_loss[tmp_plane_index] = True

    # Set the status bits for priviously incomplete frames
    if build_all_events:
        for tmp_plane_index, tmp_plane_id in enumerate(analyze_m26_header_ids):
            index = telescope_data_index
            while index >= 0:
                if telescope_data[index]['plane'] == tmp_plane_id:
                    if telescope_data[index]['frame_id'] > last_completed_m26_frame_ids[tmp_plane_index]:
                        telescope_data[index]['frame_status'] |= DATA_ERROR
                    else:
                        break
                index -= 1

    return trigger_data, trigger_data_index, telescope_data, telescope_data_index, m26_frame_ids, m26_frame_length, m26_data_loss, m26_word_index, m26_timestamps, last_m26_timestamps, m26_n_words, m26_rows, m26_frame_status, last_completed_m26_frame_ids, event_number, trigger_number, trigger_timestamp


@njit(locals={'hits_index': numba.int64, 'curr_trigger_data_index': numba.int64, 'curr_telescope_data_index': numba.int64})
def _build_events(trigger_data, trigger_data_index, telescope_data, telescope_data_index, hits, hits_index, last_completed_m26_frame_ids, timing_offset, build_all_events, analyze_m26_header_ids, plane_id_to_index):
    ''' This function is builds events from the temporary trigger and telescope data arrays.

    Parameters:
    -----------
    TBD
    '''
    latest_trigger_data_index = -1
    finished_telescope_data_indices = -1 * np.ones(shape=len(analyze_m26_header_ids), dtype=np.int64)
    last_event_trigger_data_indices = -1 * np.ones(shape=len(analyze_m26_header_ids), dtype=np.int64)
    finished_event = np.ones(shape=len(analyze_m26_header_ids), dtype=np.bool_)
    curr_event_status = np.zeros(shape=len(analyze_m26_header_ids), dtype=np.uint32)

    curr_trigger_data_index = 0
    curr_hits_index = hits_index
    # adding hits
    while curr_trigger_data_index <= trigger_data_index and np.all(finished_event):
        trigger_event_number = trigger_data[curr_trigger_data_index]['event_number']
        trigger_number = trigger_data[curr_trigger_data_index]['trigger_number']
        trigger_timestamp = trigger_data[curr_trigger_data_index]['trigger_time_stamp']
        trigger_status = trigger_data[curr_trigger_data_index]['trigger_status']
        curr_telescope_data_index = np.min(finished_telescope_data_indices) + 1
        # Reset status
        for tmp_plane_index, _ in enumerate(analyze_m26_header_ids):
            finished_event[tmp_plane_index] = False
            curr_event_status[tmp_plane_index] = 0
        while curr_telescope_data_index <= telescope_data_index:
            curr_plane_id = telescope_data[curr_telescope_data_index]['plane']
            curr_plane_index = plane_id_to_index[curr_plane_id]
            curr_frame_id = telescope_data[curr_telescope_data_index]['frame_id']
            if not finished_event[curr_plane_index] and (build_all_events or curr_frame_id <= last_completed_m26_frame_ids[curr_plane_index]):
                hit_timestamp_start = telescope_data[curr_telescope_data_index]['time_stamp'] + telescope_data[curr_telescope_data_index]['row'] * ROW_UNIT_CYCLE - 2 * FRAME_UNIT_CYCLE - timing_offset
                hit_timestamp_stop = hit_timestamp_start + FRAME_UNIT_CYCLE + ROW_UNIT_CYCLE
                if hit_timestamp_start <= trigger_timestamp and trigger_timestamp < hit_timestamp_stop:
                    curr_hits_index += 1
                    # extend hits array if neccessary
                    if curr_hits_index >= hits.shape[0]:
                        hits_tmp = np.zeros(shape=telescope_data.shape[0], dtype=hits_dtype)
                        hits = np.concatenate((hits, hits_tmp))
                    # Adding hits to event
                    hits[curr_hits_index]['plane'] = curr_plane_id
                    hits[curr_hits_index]['event_number'] = trigger_event_number
                    hits[curr_hits_index]['trigger_number'] = trigger_number
                    hits[curr_hits_index]['trigger_time_stamp'] = trigger_timestamp
                    hits[curr_hits_index]['row_time_stamp'] = hit_timestamp_start
                    hits[curr_hits_index]['frame_id'] = curr_frame_id
                    hits[curr_hits_index]['column'] = telescope_data[curr_telescope_data_index]['column']
                    hits[curr_hits_index]['row'] = telescope_data[curr_telescope_data_index]['row']
                    hits[curr_hits_index]['event_status'] = 0
                    curr_event_status[curr_plane_index] |= telescope_data[curr_telescope_data_index]['frame_status'] | trigger_status
                elif hit_timestamp_start > trigger_timestamp:
                    # latest_trigger_data_indices[plane_id_to_index[telescope_data[curr_telescope_data_index]['plane']]] = curr_trigger_data_index
                    finished_event[plane_id_to_index[telescope_data[curr_telescope_data_index]['plane']]] = True
                    if np.all(finished_event):
                        latest_trigger_data_index = curr_trigger_data_index
                        for tmp_plane_index, _ in enumerate(analyze_m26_header_ids):
                            last_event_trigger_data_indices[tmp_plane_index] = finished_telescope_data_indices[tmp_plane_index]
                        hits_index = curr_hits_index
                        # Set event status for complete event
                        index = curr_hits_index
                        while trigger_event_number == hits[index]['event_number'] and index >= 0:
                            hits[index]['event_status'] = curr_event_status[plane_id_to_index[hits[index]['plane']]]
                            index -= 1
                        break
                else:  # trigger_timestamp >= hit_timestamp_stop
                    finished_telescope_data_indices[plane_id_to_index[telescope_data[curr_telescope_data_index]['plane']]] = curr_telescope_data_index
            curr_telescope_data_index += 1
        # special case
        if build_all_events:
            hits_index = curr_hits_index
            # Set event status for complete event
            index = curr_hits_index
            while index >= 0:
                if hits[index]['event_number'] == trigger_event_number:
                    hits[index]['event_status'] = curr_event_status[plane_id_to_index[hits[index]['plane']]]
                elif hits[index]['event_number'] < trigger_event_number:
                    break
                index -= 1
            for tmp_plane_index, _ in enumerate(analyze_m26_header_ids):
                finished_event[tmp_plane_index] = True
        curr_trigger_data_index += 1

    if build_all_events:
        telescope_data_start_index = telescope_data_index + 1
    else:
        telescope_data_start_index = np.min(last_event_trigger_data_indices) + 1
    telescope_data = telescope_data[telescope_data_start_index:]
    telescope_data_index -= telescope_data_start_index
    if build_all_events:
        trigger_data_start_index = trigger_data_index + 1
    else:
        trigger_data_start_index = latest_trigger_data_index + 1
    trigger_data = trigger_data[trigger_data_start_index:]
    trigger_data_index -= trigger_data_start_index

    return trigger_data, trigger_data_index, telescope_data, telescope_data_index, hits, hits_index
