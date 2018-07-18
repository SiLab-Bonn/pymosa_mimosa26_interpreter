''' Class to convert Mimosa26 raw data.

General structure of Mimosa26 raw data:
 - raw data is 32 bit: first 8 bits are always 20 (hex) for M26
 - next four bits are plane number
 - last 4 bits of 16 high bits are not used
 - rest of 16 bit is data word

General chain of Mimosa26 raw data words:
 - Frame header containing timestamp (2x, HIGH + LOW) [word index 0 + 1]
 - Frame number (2x, HIGH + LOW) [word index 2 + 3]
 - Length of data (2x, HIGH + LOW) [word index 4 + 5]
 - Hit data (2x)
 - ...
 - ...
 - Tailer (2x, HIGH + LOW) [word index 6 + 7]

'''
import numba
from numba import njit
import numpy as np


FRAME_UNIT_CYCLE = 4608  # time for one frame in units of 40 MHz clock cylces (115.2 * 40)

hit_dtype = np.dtype([('plane', '<u1'), ('frame', '<u4'), ('time_stamp', '<u4'), ('trigger_number', '<u2'),
                      ('column', '<u2'), ('row', '<u2'), ('event_status', '<u4')])
tlu_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<i4'), ('frame', '<u4')])

# Event error codes
# NO_ERROR = 0  # No error
MULTIPLE_TRG_WORD = 0x00000001  # Event has more than one trigger word
NO_TRG_WORD = 0x00000002  # Some hits of the event have no trigger word
DATA_ERROR = 0x00000004  # Event has data word combinations that does not make sense (tailor at wrong position, not increasing frame counter ...)
EVENT_INCOMPLETE = 0x00000008  # Data words are missing (e.g. tailor header)
UNKNOWN_WORD = 0x00000010  # Event has unknown words
UNEVEN_EVENT = 0x00000020  # Event has uneven amount of data words
TRG_ERROR = 0x00000040  # A trigger error occurred
TRUNC_EVENT = 0x00000080  # Event had too many hits and was truncated
TAILER_H_ERROR = 0x00000100  # tailer high error
TAILER_L_ERROR = 0x00000200  # tailer low error
MIMOSA_OVERFLOW = 0x00000400  # mimosa overflow
NO_HIT = 0x00000800  # events without any hit, useful for trigger number debugging
COL_ERROR = 0x00001000  # column number error
ROW_ERROR = 0x00002000  # row number error
TRG_WORD = 0x00004000  # column number overflow
TS_OVERFLOW = 0x00008000  # timestamp overflow


@njit
def is_mimosa_data(word):  # Check for Mimosa data word
    return 0xFF000000 & word == 0x20000000


@njit
def is_data_loss(word):
    return 0x00020000 & word == 0x20000


@njit
def get_plane_number(word):  # There are 6 planes in the stream, starting from 1; return plane number
    return (word >> 20) & 0xF


@njit
def get_frame_id_high(word):  # Get the frame id from the frame id high word
    return 0x0000FFFF & word


@njit
def get_frame_id_low(word):  # Get the frame id from the frame id low word
    return (0x0000FFFF & word) << 16


@njit
def is_frame_header(word):  # Check if frame header high word
    return 0x00010000 & word == 0x10000


@njit
def is_frame_tailer_high(word):  # Check if frame tailer high word
    return word & 0xFFFF == 0xaa50


@njit
def is_frame_tailer_low(word, plane):  # Check if frame header low word for the actual plane
    return (word & 0xFFFF) == (0xaa50 | plane)


@njit
def get_frame_length(word):
    return (word & 0xFFFF) * 2


@njit
def get_n_hits(word):  # Returns the number of hits given by actual column word
    return word & 0x3


@njit
def get_column(word):
    return (word >> 2) & 0x7FF


@njit
def get_row(word):
    return (word >> 4) & 0x7FF


@njit
def get_m26_timestamp_high(word):  # time stamp of Mimosa26 data from frame header high
    return word & 0x0000FFFF


@njit
def get_m26_timestamp_low(word):  # time stamp of Mimosa26 data from frame header low
    return (0x0000FFFF & word) << 16


@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return word & 0xF


@njit
def has_overflow(word):
    return word & 0x00008000 != 0


@njit
def is_trigger_word(word):
    return 0x80000000 & word == 0x80000000


@njit
def get_trigger_timestamp(word):  # time stamp of TLU word
    return (word & 0x7FFF0000) >> 16


@njit
def get_trigger_number(word, trigger_data_format):  # trigger number of TLU word
    if trigger_data_format == 2:
        return word & 0x0000FFFF
    else:
        return word & 0x7FFFFFFF


@njit
def add_event_status(plane_id, event_status, status_code):
    event_status[plane_id] |= status_code


@njit(locals={'last_timestamp': numba.uint32})
def build_hits(raw_data, frame_id, last_frame_id, frame_length, word_index, n_words, row, event_status,
               event_number, trigger_number, timestamp, last_timestamp, max_hits_per_chunk, trigger_data_format=2):
    ''' Main interpretation function. Loops over the raw data and creates a hit array. Data errors are checked for.
    A lot of parameters are needed, since the variables have to be buffered for chunked analysis and given for
    each call of this function.

    Parameters:
    -----------
    raw_data : np.array
        The array with the raw data words
    frame_id : np.array, shape 6
        The counter value of the actual frame for each plane, 0 if not set
    last_frame_id : np.array, shape 6
        The counter value of the last frame for each plane, -1 if not available
    frame_length : np.array, shape 6
        The number of data words in the actual frame frame for each plane, 0 if not set
    word_index : np.array, shape 6
        The word index of the actual frame for each plane, 0 if not set
    n_words : np.array, shape 6
        The number of words containing column / row info for each plane, 0 if not set
    row : np.array, shape 6
        The actual readout row (rolling shutter) for each plane, 0 if not set
    event_status : np.array
        Actual event status for each plane
    event_number : np.array, shape 6
        The event counter set by the software counting full events for each plane
    trigger_number : np.array
        The trigger number of the actual event
    timestamp : np.array shape 7
        First index contains the trigger timestamp. Last six indices contains the timestamps from Mimosa26 frames
    last_timestamp : np.array
        Timestamp of last Mimosa26 frame
    max_hits_per_chunk : number
        Maximum expected hits per chunk. Needed to allocate hit array.
    trigger_data_format : integer
        Number which indicates the used trigger data format.
        0: TLU word is trigger number (not supported)
        1: TLU word is timestamp (not supported)
        2: TLU word is 15 bit timestamp + 16 bit trigger number
        Only trigger data format 2 is supported, since the event building requires a trigger timestamp in order to work reliably.

    Returns
    -------
    A list of all input parameters.
    '''
    # The raw data order of the Mimosa 26 data should be always START / FRAMEs ID / FRAME LENGTH / DATA
    # Since the clock is the same for each plane; the order is START plane 1, START plane 2, ...

    hits = np.empty(shape=(max_hits_per_chunk,), dtype=hit_dtype)  # Result hits array
    hit_index = 0  # Pointer to actual hit in resul hit arrray; needed to append hits every event

    # Loop over raw data words
    for raw_i in range(raw_data.shape[0]):
        word = raw_data[raw_i]  # Actual raw data word
        if is_mimosa_data(word):  # Check if word is from Mimosa26. Other words can come from TLU.
            # Check to which plane the data belongs
            plane_id = get_plane_number(word) - 1  # The actual_plane if the actual word belongs to (0 to 5)
            # Interpret the word of the actual plane
            if is_data_loss(word):
                # Reset word index for all planes
                for i in range(6):
                    word_index[i] = -1
            elif is_frame_header(word):  # New event for actual plane; events are aligned at this header
                if plane_id == 0:
                    last_timestamp = timestamp[1]  # Timestamp of last Mimosa26 frame
                    last_frame_id = frame_id[1]  # Last Mimosa26 frame number
                # Get Mimosa26 timestamp from header high word
                timestamp[plane_id + 1] = get_m26_timestamp_high(word) | (timestamp[plane_id + 1] & 0xFFFF0000)
                word_index[plane_id] = 0
            elif word_index[plane_id] == -1:  # Trash data
                # TODO: add event status trash data
                pass
            else:  # Correct M26 data
                word_index[plane_id] += 1
                if word_index[plane_id] == 1:  # After header high word header low word comes
                    # TODO: make this nicer
                    if (timestamp[plane_id + 1] >> 16) != (0x0000FFFF & word) and (((timestamp[plane_id + 1] >> 16) + 1) & 0xFFFF) != (0x0000FFFF & word):  # Timestamp has overflow
                        add_event_status(plane_id + 1, event_status, TS_OVERFLOW)

                    # Get Mimosa26 timestamp from header low word
                    timestamp[plane_id + 1] = get_m26_timestamp_low(word) | timestamp[plane_id + 1] & 0x0000FFFF

                elif word_index[plane_id] == 2:  # Next word should be the frame ID high word
                    frame_id[plane_id + 1] = get_frame_id_high(word) | (frame_id[plane_id + 1] & 0xFFFF0000)

                elif word_index[plane_id] == 3:  # Next word should be the frame ID low word
                    frame_id[plane_id + 1] = get_frame_id_low(word) | (frame_id[plane_id + 1] & 0x0000FFFF)

                elif word_index[plane_id] == 4:  # Next word should be the frame length high word
                    frame_length[plane_id] = get_frame_length(word)

                elif word_index[plane_id] == 5:  # Next word should be the frame length low word (=high word, one data line, the number of words is repeated 2 times)
                    if frame_length[plane_id] != get_frame_length(word):
                        add_event_status(plane_id + 1, event_status, EVENT_INCOMPLETE)

                elif word_index[plane_id] == 6 + frame_length[plane_id]:  # Next word should be the frame tailer high word
                    if not is_frame_tailer_high(word):
                        add_event_status(plane_id + 1, event_status, TAILER_H_ERROR)

                elif word_index[plane_id] == 7 + frame_length[plane_id]:  # Last word should be the frame tailer low word
                    frame_length[plane_id] = -1
                    n_words[plane_id] = 0
                    if not is_frame_tailer_low(word, plane=plane_id + 1):
                        add_event_status(plane_id + 1, event_status, TAILER_L_ERROR)

                else:  # Column / Row words (actual data word with hits)
                    if n_words[plane_id] == 0:  # First word contains the row info and the number of data words for this row
                        if word_index[plane_id] == 6 + frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                            add_event_status(plane_id + 1, event_status, UNEVEN_EVENT)
                        else:
                            n_words[plane_id] = get_n_words(word)
                            row[plane_id] = get_row(word)  # Get row from data word
                        if has_overflow(word):
                            add_event_status(plane_id + 1, event_status, MIMOSA_OVERFLOW)
                            n_words[plane_id] = 0
                        if row[plane_id] > 576:  # Row overflow
                            add_event_status(plane_id + 1, event_status, ROW_ERROR)
                    else:
                        n_words[plane_id] = n_words[plane_id] - 1  # Count down the words
                        n_hits = get_n_hits(word)
                        column = get_column(word)  # Get column from data word
                        if column >= 1152:  # Column overflow
                            add_event_status(plane_id + 1, event_status, COL_ERROR)
                        for k in range(n_hits + 1):
                            if hit_index >= hits.shape[0]:
                                hits_tmp = np.empty(shape=(max_hits_per_chunk,), dtype=hit_dtype)
                                hits = np.concatenate((hits, hits_tmp))
                            # Store hits
                            hits[hit_index]['frame'] = frame_id[plane_id + 1]
                            hits[hit_index]['plane'] = plane_id + 1
                            hits[hit_index]['time_stamp'] = timestamp[plane_id + 1]
                            hits[hit_index]['trigger_number'] = trigger_number
                            hits[hit_index]['column'] = column + k
                            hits[hit_index]['row'] = row[plane_id]
                            hits[hit_index]['event_status'] = event_status[plane_id + 1]
                            hit_index = hit_index + 1

                        # Reset event status
                        for i in range(1, 7):
                            event_status[i] = 0
        elif is_trigger_word(word):  # Raw data word is TLU word
            trigger_number = get_trigger_number(word, trigger_data_format)
            # Calculating 31bit timestamp from 15bit trigger timestamp
            # and use last_timestamp (frame header timestamp) for that.
            # Assumption: last_timestamp is updated more frequent than
            # the 15bit trigger timestamp can overflow. The frame is occurring
            # every 4608 clock cycles (115.2 us).
            timestamp[0] = get_trigger_timestamp(word) | (last_timestamp & 0xFFFF8000)
            # Check if trigger timestamp overflow (15bit) has occurred
            # and add the length of the trigger timstamp counter
            if timestamp[0] < last_timestamp:
                timestamp[0] = timestamp[0] + 2**15
            # Check for timestamp overflow (32bit) and calculate timestamp
            # difference between start of frame and trigger
            if timestamp[0] < last_timestamp:
                delta_timestamp = timestamp[0] + (2**32 - last_timestamp)
            else:
                delta_timestamp = timestamp[0] - last_timestamp
            # Get actual frame number where trigger word occured:
            # This is last frame number (relative to where 31 bit trigger timestamp is calculated) + offset between last timestamp
            # and 31 bit trigger timestamp in units of full frame cycles.
            frame_id[0] = last_frame_id + np.floor_divide(delta_timestamp, FRAME_UNIT_CYCLE)
            if hit_index >= hits.shape[0]:
                hits_tmp = np.empty(shape=(max_hits_per_chunk,), dtype=hit_dtype)
                hits = np.concatenate((hits, hits_tmp))
            hits[hit_index]['frame'] = frame_id[0]  # Frame number of trigger timestamp (aligned to Mimosa26 frame)
            hits[hit_index]['plane'] = 255  # TLU data is indicated with this plane number
            hits[hit_index]['time_stamp'] = timestamp[0]  # Timestamp of TLU word
            hits[hit_index]['trigger_number'] = trigger_number
            hits[hit_index]['column'] = 0
            hits[hit_index]['row'] = delta_timestamp % FRAME_UNIT_CYCLE  # Distance between trigger timestamp to timestamp of last Mimosa26 frame
            hits[hit_index]['event_status'] = event_status[0]  # event status of TLU
            hit_index = hit_index + 1
            add_event_status(0, event_status, TRG_WORD)
        else:  # Raw data contains unknown word, neither M26 nor TLU word
            add_event_status(0, event_status, UNKNOWN_WORD)
    return (hits[:hit_index], frame_id, last_frame_id, frame_length, word_index, n_words, row,
            event_status, event_number, trigger_number, timestamp, last_timestamp)


class RawDataInterpreter(object):
    ''' Class to convert the raw data chunks to hits'''

    def __init__(self, max_hits_per_chunk=500000, trigger_data_format=2):
        self.max_hits_per_chunk = max_hits_per_chunk
        self.trigger_data_format = trigger_data_format
        self.reset()

    def reset(self):  # Reset variables
        # Per frame variables
        self.frame_id = np.zeros(shape=(7, ), dtype=np.uint32)  # The counter value of the actual frame, 6 Mimosa planes + TLU
        self.last_frame_id = self.frame_id[1]
        self.frame_length = np.ones(shape=(6, ), dtype=np.int32) * -1  # The number of data words in the actual frame
        self.word_index = np.zeros(shape=(6, ), dtype=np.int32)  # The word index per device of the actual frame
        self.timestamp = np.zeros(shape=(7, ), dtype=np.uint32)  # The timestamp for each plane (in units of 40 MHz), first index corresponds to TLU word timestamp, last 6 indices are timestamps of M26 frames
        self.last_timestamp = self.timestamp[1]
        self.n_words = np.zeros(shape=(6, ), dtype=np.uint32)  # The number of words containing column / row info
        self.row = np.ones(shape=(6, ), dtype=np.int32) * -1  # The actual readout row (rolling shutter)

        # Per event variables
        self.tlu_word_index = np.zeros(shape=(6, ), dtype=np.uint32)  # TLU buffer index for each plane; needed to append hits
        self.event_status = np.zeros(shape=(7, ), dtype=np.uint32)  # Actual event status for each plane, TLU and 6 Mimosa planes
        self.event_number = np.ones(shape=(6, ), dtype=np.int64) * -1  # The event counter set by the software counting full events for each plane
        self.trigger_number = 0  # The trigger number of the actual event

    def interpret_raw_data(self, raw_data):
        chunk_result = build_hits(raw_data=raw_data,
                                  frame_id=self.frame_id,
                                  last_frame_id=self.last_frame_id,
                                  frame_length=self.frame_length,
                                  word_index=self.word_index,
                                  n_words=self.n_words,
                                  row=self.row,
                                  event_status=self.event_status,
                                  event_number=self.event_number,
                                  trigger_number=self.trigger_number,
                                  timestamp=self.timestamp,
                                  last_timestamp=self.last_timestamp,
                                  max_hits_per_chunk=self.max_hits_per_chunk,
                                  trigger_data_format=self.trigger_data_format)

        # Set updated buffer variables
        (hits,
         self.frame_id,
         self.last_frame_id,
         self.frame_length,
         self.word_index,
         self.n_words,
         self.row,
         self.event_status,
         self.event_number,
         self.trigger_number,
         self.timestamp,
         self.last_timestamp) = chunk_result

        return hits
