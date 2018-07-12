''' Class to convert Mimosa26 raw data.

General structure of Mimosa26 raw data:
 - raw data is 32 bit: first 8 bits are always 20 (hex) for M26
 - next four bits are plane number
 - last 4 bits of 16 high bits are not used
 - rest of 16 bit is data word

General chain of Mimosa26 raw data words
 - timestamp [word index 1]
 - frame number (2x, HIGH + LOW) [word index 2 + 3]
 - length of data (2x, HIGH + LOW) [word index 4 + 5]
 - hit data (2x)
 - ...
 - ...
 - tailer (2x, HIGH + LOW) [word index 6 + 7]

'''
from numba import njit
import numpy as np

FRAME_UNIT_CYCLE = 4608  # time for one frame in units of 40 MHz clock cylces

hit_dtype = np.dtype([('plane', '<u1'), ('frame', '<u4'), ('time_stamp', '<u4'), ('trigger_number', '<u2'),
                      ('column', '<u2'), ('row', '<u2'), ('event_status', '<u2')])
tlu_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<i4'), ('frame', '<u4')])

# Event error codes
# NO_ERROR = 0  # No error
MULTIPLE_TRG_WORD = 0x00000001  # Event has more than one trigger word
NO_TRG_WORD = 0x00000002  # Some hits of the event have no trigger word
DATA_ERROR = 0x00000004  # Event has data word combinations that does not make sense (tailor at wrong position, not increasing frame counter ...)
EVENT_INCOMPLETE = 0x00000008  # Data words are missing (e.g. tailor header)
UNKNOWN_WORD = 0x00000010  # Event has unknown words
UNEVEN_EVENT = 0x00000020  # Event has uneven amount of data words
TRG_ERROR = 0x00000040  # A trigger error occured
TRUNC_EVENT = 0x00000080  # Event had to many hits and was truncated
TAILER_H_ERROR = 0x00000100  # tailer high error
TAILER_L_ERROR = 0x00000200  # tailer low error
MIMOSA_OVERFLOW = 0x00000400  # mimosa overflow
NO_HIT = 0x00000800  # events without any hit, usefull for trigger number debugging
COL_ERROR = 0x00001000  # column number error
ROW_ERROR = 0x00002000  # row number error
TRG_WORD = 0x00004000  # column number overflow
TS_OVERFLOW = 0x00008000  # timestamp overflow


@njit
def is_mimosa_data(word):  # Check for Mimosa data word
    return 0xFF000000 & word == 0x20000000


@njit
def get_plane_number(word):  # There are 6 planes in the stream, starting from 1; return plane number
    return (word >> 20) & 0xF


@njit
def is_data_loss(word):
    return 0x00020000 & word == 0x20000


@njit
def get_frame_id_high(word):  # Get the frame id from the frame id high word
    return 0x0000FFFF & word


@njit
def get_frame_id_low(word):  # Get the frame id from the frame id low word
    return (0x0000FFFF & word) << 16


@njit
def is_frame_header_high(word):  # Check if frame header high word
    return 0x000F0000 & word == 0x10000


@njit
def is_frame_tailer_high(word):  # Check if frame header high word
    return word & 0xFFFF == 0xaa50


@njit
def is_frame_tailer_low(word, plane):  # Check if frame header low word for the actual plane
    return (word & 0xFFFF) == (0xaa50 | plane)


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
def is_trigger_word(word):
    return 0x80000000 & word == 0x80000000


@njit
def get_frame_length(word):
    return (word & 0xFFFF) * 2


@njit
def get_trigger_timestamp(word):  # time stamp of TLU word
    return (word & 0x7FFF0000) >> 16


@njit
def get_trigger_number(word, trigger_data_format):  # trigger number of TLU word
    if trigger_data_format == 2:
        return word & np.uint16(0xFFFF)
    else:
        return word & 0x7FFFFFFF


@njit
def get_timestamp_high(word):  # time stamp high of Mimosa26 data
    return (0x0000FFFF & word) << 16


@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return word & 0xF


@njit
def has_overflow(word):
    return word & 0x00008000 != 0


@njit
def add_event_status(plane_id, event_status, status_code):
    event_status[plane_id] |= status_code


@njit
def build_hits(raw_data, frame_id, last_frame_id, frame_length, word_index, n_words, row, event_status,
               event_number, trigger_number, timestamp, last_timestamp, max_hits_per_chunk, trigger_data_format=2):
    ''' Main interpretation function. Loops over the raw data and creates a hit array. Data errors are checked for.
    A lot of parameters are needed, since the variables have to be buffered for chunked analysis and given for
    each call of this function.

    Parameters:
    ----------
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

    hits = np.zeros(shape=(max_hits_per_chunk,), dtype=hit_dtype)  # Result hits array
    hit_index = 0  # Pointer to actual hit in resul hit arrray; needed to append hits every event

    # Loop over raw data words
    for raw_i in range(raw_data.shape[0]):
        word = raw_data[raw_i]  # Actual raw data word
        if is_mimosa_data(word):  # Check if word is from M26. Other words can come from TLU.
            # Check to which plane the data belongs
            plane_id = get_plane_number(word) - 1  # The actual_plane if the actual word belongs to (0 .. 5)
            # Interpret the word of the actual plane
            if is_data_loss(word):
                # Reset word index and event status for all planes
                # Note: event_status[0] is TLU event status
                word_index[0] = -1
                event_status[1] = 0
                word_index[1] = -1
                event_status[2] = 0
                word_index[2] = -1
                event_status[3] = 0
                word_index[3] = -1
                event_status[4] = 0
                word_index[4] = -1
                event_status[5] = 0
                word_index[5] = -1
                event_status[6] = 0
            elif is_frame_header_high(word):  # New event for actual plane; events are aligned at this header
                if plane_id == 0:
                    last_timestamp = timestamp[1]  # timestamp of last M26 frame
                    last_frame_id = frame_id[1]  # last M26 frame number
                # TODO: what is this
                timestamp[plane_id + 1] = (timestamp[plane_id + 1] & 0xFFFF0000) | word & 0xFFFF
                word_index[plane_id] = 0
            elif word_index[plane_id] == -1:  # trash data
                # TODO: add event status trash data
                pass
            else:  # correct M26 data
                word_index[plane_id] += 1
                if word_index[plane_id] == 1:  # 1. timestamp high
                    # TODO: make this nicer
                    if (timestamp[plane_id + 1] >> 16) != (0x0000FFFF & word) and (((timestamp[plane_id + 1] >> 16) + 1) & 0xFFFF) != (0x0000FFFF & word):
                        add_event_status(plane_id + 1, event_status, TS_OVERFLOW)

                    timestamp[plane_id + 1] = get_timestamp_high(word) | timestamp[plane_id + 1] & 0x0000FFFF  # TODO: this is Mimosa26 timestamp?

                elif word_index[plane_id] == 2:  # 2. word should have the frame ID high word
                    frame_id[plane_id + 1] = get_frame_id_high(word) | (frame_id[plane_id + 1] & 0xFFFF0000)  # TODO: make this nicer

                elif word_index[plane_id] == 3:  # 3. word should have the frame ID low word
                    frame_id[plane_id + 1] = get_frame_id_low(word) | (frame_id[plane_id + 1] & 0x0000FFFF)

                elif word_index[plane_id] == 4:  # 4. word should have the frame length high word
                    frame_length[plane_id] = get_frame_length(word)

                elif word_index[plane_id] == 5:  # 5. word should have the frame length low word (=high word, one data line, the number of words is repeated 2 times)
                    if frame_length[plane_id] != get_frame_length(word):
                        add_event_status(plane_id + 1, event_status, EVENT_INCOMPLETE)

                elif word_index[plane_id] == 6 + frame_length[plane_id]:  # Second last word is frame tailer high word
                    if not is_frame_tailer_high(word):
                        add_event_status(plane_id + 1, event_status, TAILER_H_ERROR)

                elif word_index[plane_id] == 7 + frame_length[plane_id]:  # First last word is frame tailer low word
                    frame_length[plane_id] = -1
                    n_words[plane_id] = 0
                    if not is_frame_tailer_low(word, plane=plane_id + 1):
                        add_event_status(plane_id + 1, event_status, TAILER_L_ERROR)

                else:  # Column / Row words (actual data word)
                    if n_words[plane_id] == 0:  # First word contains the row info and the number of data words for this row
                        if word_index[plane_id] == 6 + frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                            add_event_status(plane_id + 1, event_status, UNEVEN_EVENT)
                        else:
                            n_words[plane_id] = get_n_words(word)
                            row[plane_id] = get_row(word)  # get row from data word
                        if has_overflow(word):
                            add_event_status(plane_id + 1, event_status, MIMOSA_OVERFLOW)
                            n_words[plane_id] = 0
                        if row[plane_id] > 576:  # row overflow
                            add_event_status(plane_id + 1, event_status, ROW_ERROR)
                    else:
                        n_words[plane_id] = n_words[plane_id] - 1  # Count down the words
                        n_hits = get_n_hits(word)
                        column = get_column(word)  # get column from data word
                        if column >= 1152:  # column overflow
                            add_event_status(plane_id + 1, event_status, COL_ERROR)
                        for k in range(n_hits + 1):
                            if hit_index < max_hits_per_chunk:
                                # store hits
                                hits[hit_index]['frame'] = frame_id[plane_id + 1]
                                hits[hit_index]['plane'] = plane_id + 1
                                hits[hit_index]['time_stamp'] = timestamp[plane_id + 1]
                                hits[hit_index]['trigger_number'] = trigger_number
                                hits[hit_index]['column'] = column + k
                                hits[hit_index]['row'] = row[plane_id]
                                hits[hit_index]['event_status'] = event_status[plane_id + 1]
                                hit_index = hit_index + 1
                            else:
                                # truncated data
                                add_event_status(plane_id + 1, event_status, TRUNC_EVENT)
        elif is_trigger_word(word):  # raw data word is TLU word
            trigger_number = get_trigger_number(word, trigger_data_format)
            # TODO: what is this? make this nicer
            timestamp[0] = get_trigger_timestamp(word) | np.uint32(last_timestamp & 0xFFFF8000)
            tlu_flag = 0
            # TODO: what is this?
            if (timestamp[0] - last_timestamp) & 0x8000 == 0x8000:  # if timestamp < ts_pre
                timestamp[0] = timestamp[0] + np.uint32(0x8000)
                tlu_flag = 1

            # TODO: make this nicer, what is this??
            frame_id[0] = last_frame_id + ((timestamp[0] - last_timestamp) & 0x7FFF) / FRAME_UNIT_CYCLE  # artificial frame number (aligned to M26 frames) of TLU word
            hits[hit_index]['frame'] = frame_id[0]
            hits[hit_index]['plane'] = 255  # TLU data is indicated with this plane number
            hits[hit_index]['time_stamp'] = timestamp[0]  # timestamp of TLU word
            hits[hit_index]['trigger_number'] = trigger_number
            hits[hit_index]['column'] = tlu_flag
            hits[hit_index]['row'] = np.uint16(((timestamp[0] - last_timestamp) & 0x7FFF) % FRAME_UNIT_CYCLE)  # number of clock cycles between TLU word timestamp and timestamp of last M26 frame in units of full FRAME_UNIT_CYCLE
            hits[hit_index]['event_status'] = event_status[0]
            hit_index = hit_index + 1
            # TODO: fix event status, something is wrong with overall event status
            add_event_status(0, event_status, TRG_WORD)
        else:
            add_event_status(0, event_status, UNKNOWN_WORD)
    return (hits[:hit_index], frame_id, last_frame_id, frame_length, word_index, n_words, row,
            event_status, event_number, trigger_number, timestamp, last_timestamp)


class RawDataInterpreter(object):
    ''' Class to convert the raw data chunks to hits'''

    def __init__(self, max_hits_per_chunk=5000000, trigger_data_format=2):
        self.max_hits_per_chunk = max_hits_per_chunk
        self.trigger_data_format = trigger_data_format
        self.reset()

    def reset(self):  # Reset variables
        # Per frame variables
        self.frame_id = np.zeros(7, np.uint32)  # The counter value of the actual frame, 6 Mimosa planes + TLU
        self.last_frame_id = self.frame_id[1]
        self.frame_length = np.ones(6, np.int32) * -1  # The number of data words in the actual frame
        self.word_index = np.zeros(6, np.int32) * -1  # The word index per device of the actual frame
        self.timestamp = np.zeros(7, np.uint32)  # The timestamp for each plane (in units of 40 MHz), first index corresponds to TLU word timestamp, last 6 indices are timestamps of M26 frames
        self.last_timestamp = self.timestamp[1]
        self.n_words = np.zeros(6, np.uint32)  # The number of words containing column / row info
        self.row = np.ones(6, np.int32) * -1  # the actual readout row (rolling shutter)

        # Per event variables
        self.tlu_word_index = np.zeros(6, np.uint32)  # TLU buffer index for each plane; needed to append hits
        self.event_status = np.zeros(shape=(7, ), dtype=np.uint16)  # Actual event status for each plane, TLU and 6 Mimosa planes
        self.event_number = np.ones(6, np.int64) * -1  # The event counter set by the software counting full events for each plane
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
