''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.

General structure of Mimosa26 raw data:
 - raw data is 32 bit: first 8 bits are always 20 (hex) for M26
 - next for bits are plane number
 - last 4 bits of 16 high bits are not used
 - rest of 16 bit is data word

General chain of Mimosa26 raw data words
 - timestamp (2x?), word index 1
 - frame number (2x, HIGH + LOW), word index 2, 3
 - length of data (2x, HIGH + LOW), word index 4, 5
 - data (2x)
 - ...
 - ...
 - tailer (2x, HIGH + LOW), word index 6, 7

'''
from numba import njit
import numpy as np

hit_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<i4'), ('timestamp', '<u4'), ('trigger_number_begin', '<u2'),('trigger_number_end', '<u2'), 
                      ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'),('trigger_status', '<u2'), ('event_status', '<u2')])

# Event error codes
NO_ERROR = 0  # No error
MULTIPLE_TRG_WORD = 1  # Event has more than one trigger word
NO_TRG_WORD = 2  # Some hits of the event have no trigger word
DATA_ERROR = 4  # Event has data word combinations that does not make sense (tailor at wrong position, not increasing frame counter ...)
EVENT_INCOMPLETE = 8  # Data words are missing (e.g. tailor header)
UNKNOWN_WORD = 16  # Event has unknown words
UNEVEN_EVENT = 32  # Event has uneven amount of data words
TRG_ERROR = 64  # A trigger error occured
TRUNC_EVENT = 128  # Event had to many hits and was truncated
TDC_WORD = 256  # Event has a TDC word
MANY_TDC_WORDS = 512  # Event has more than one valid TDC word
TDC_OVERFLOW = 1024  # Event has TDC word indicating a TDC overflow
NO_HIT = 2048  # events without any hit, usefull for trigger number debugging
COL_OVERFLOW = 4096  # column number overflow
ROW_OVERFLOW = 8192  # row number overflow


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
def get_data_length(word):
    return (word & 0x0000FFFF) * 2


@njit
def is_overflow(word):
    return word & 0x8000 == 0x8000


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
def get_trigger_word(word, trigger_data_format):  # Returns trigger word (trigger id and timestamp)
    if trigger_data_format == 0:
        trigger_number = word & 0x7FFFFFFF
        timestamp = 0  # set timestamp to zero
    elif trigger_data_format == 1:
        timestamp = word & 0x7FFFFFFF
        trigger_number = 0  # set trigger number to zero
    else:
        trigger_number = word & 0xFFFF
        timestamp = (word & 0x7FFF0000) >> 16
    return trigger_number, timestamp


@njit
def get_trigger_timestamp(word):  # time stamp of trigger
    return (word & 0x7FFF0000) >> 16


@njit
def get_trigger_number(word, trigger_data_format):  # time stamp of trigger
    if trigger_data_format == 2:
        return word & 0xFFFF
    else:
        return word & 0x7FFFFFFF


@njit
def get_timestamp_low(word):  # time stamp low of Mimosa26 data
    return 0x0000FFFF & word


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
def finish_event(plane_id, hits_buffer, hit_buffer_index, event_status, hits, hit_index):  # Append buffered hits to hit object
    for i_hit in range(hit_buffer_index):  # Loop over buffered hits
        hits[hit_index] = hits_buffer[plane_id, i_hit]
        hits[hit_index]['event_status'] = event_status
        hit_index += 1
        if hit_index > hits.shape[0] - 1:
            raise RuntimeError('Hits array is too small for the hits. Tell developer!')
    return hit_index  # Return actual hit index; needed to append correctly at next call of finish_event


# @njit
# def get_event_number(plane_id, event_number, timestamp):
#     for i_ts, ts in enumerate(timestamp):
#         if i_ts != plane_id:
#             if ts == timestamp[plane_id]:
#                 return event_number[i_ts]
#             elif ts > timestamp[plane_id]:
#                 return -1
#             elif event_number[plane_id] < event_number[i_ts]:  ## and ts < timestamp[plane_id]
#                 event_number[plane_id] = event_number[i_ts]
#     return event_number[plane_id]+1


@njit
def build_hits(raw_data, frame_id, last_frame_id, frame_length, word_index, n_words, row, hits_buffer, hit_buffer_index,
               event_status, event_number, trigger_number, trigger_number_begin, trigger_number_end, timestamp, max_hits_per_event, debug, trigger_data_format=0):
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
    hits_buffer : np.array, shape 6, max_hits_per_event
        Buffers actual event hits, needed since raw data is analyzed in chunks
    hit_buffer_index  : np.array, shape 6
        Hit buffer index for each plane, needed to append hits
    event_status : np.array
        Actual event status for each plane
    event_number : np.array, shape 6
        The event counter set by the software counting full events for each plane
    trigger_number_begin : number
        The actual event trigger number recieved during the frame (begin)
    trigger_number_end : number
        The actual event trigger number recieved during the frame (end)
    timestamp : np.array shape 6
        The timestamp read from mimosa header
    max_hits_per_event : number
        Maximum expected hits per event. Needed to allocate hit buffer.
    debug : number
        1st bit: 1=write all trigger as plane 0, 0=off
        2-32: not used
    trigger_data_format : int
        0: tlu word is trigger number
        1: tlu word is timestamp
        2: tlu word is 15 bit timestamp + 16 bit trigger number

    Returns
    -------
    A list of all input parameters, but raw_data is exchanged for a hit array and max_hits_per_event is not returned.
    '''
    # The raw data order of the Mimosa 26 data should be always START / FRAMEs ID / FRAME LENGTH / DATA
    # Since the clock is the same for each plane; the order is START plane 1, START plane 2, ...

    hits = np.zeros(shape=(raw_data.shape[0] * 5), dtype=hit_dtype)  # Result hits array
    hit_index = 0  # Pointer to actual hit in resul hit arrray; needed to append hits every event

    # Loop over raw data words
    for raw_i in range(raw_data.shape[0]):
        word = raw_data[raw_i]  # Actual raw data word
        if is_mimosa_data(word):  # Check if word is from M26. Other words can come from TLU.
            # Check to which plane the data belongs
            plane_id = get_plane_number(word) - 1  # The actual_plane if the actual word belongs to (0 .. 5)

            # Interpret the word of the actual plane
            if is_data_loss(word):
                # Reset all planes
                word_index[0] = -1
                event_status[0] = 0
                word_index[1] = -1
                event_status[1] = 0
                word_index[2] = -1
                event_status[2] = 0
                word_index[3] = -1
                event_status[3] = 0
                word_index[4] = -1
                event_status[4] = 0
                word_index[5] = -1
                event_status[5] = 0
            elif is_frame_header_high(word):  # New event for actual plane; events are aligned at this header
                if event_number[plane_id] >= 0:  # First event 0 should not trigger a last event finish, since there is none
                    if last_frame_id[plane_id] > 0 and frame_id[plane_id] != last_frame_id[plane_id] + 1:
                        add_event_status(plane_id, event_status, DATA_ERROR)
                    last_frame_id[plane_id] = frame_id[plane_id]
                    # print 'Finsihed event', event_number[plane_id], 'for plane', plane_id
                    hit_index = finish_event(plane_id, hits_buffer, hit_buffer_index[plane_id], event_status[plane_id], hits, hit_index)
                # Reset counter
                hit_buffer_index[plane_id] = 0
                event_status[plane_id] = 0
                event_number[plane_id] += 1  # Increase event counter for this plane
                word_index[plane_id] = 0
                                
#                 # shift trigger_number
#                 if last_frame_id[plane_id] > 0 and frame_id[plane_id] != last_frame_id[plane_id] + 1:
#                         trigger_number_begin[2] = 0x7FFFFFFF
#                         trigger_number_end[2] = 0x7FFFFFFF
#                         trigger_number_begin[1] = 0x7FFFFFFF
#                         trigger_number_end[1] = 0x7FFFFFFF
#                         trigger_number_begin[0] = 0x7FFFFFFF
#                         trigger_number_end[0] = 0x7FFFFFFF
#                         trigger_number_begin[3] = 0x7FFFFFFF
#                         trigger_number_end[3] = 0x7FFFFFFF
#                         add_event_status(plane_id, event_status, DATA_ERROR)
#                 elif plane_id == 0:  # TODO reset trigger_number at the first header not plane_id==0
#                     trigger_number_begin[2] = trigger_number_begin[1]
#                     trigger_number_end[2] = trigger_number_end[1]
#                     trigger_number_begin[1] = trigger_number_begin[0]
#                     trigger_number_end[1] = trigger_number_end[0]
#                     trigger_number_begin[0] = 0x7FFFFFFF
#                     trigger_number_end[0] = 0x7FFFFFFF
#                     if trigger_number_begin[2] == 0x7FFFFFFF:
#                         if trigger_number_begin[1] == 0x7FFFFFFF:
#                             trigger_number_begin[3] = 0x7FFFFFFF
#                             trigger_number_end[3] = 0x7FFFFFFF
#                         else:
#                             trigger_number_begin[3] = trigger_number_begin[1]
#                             trigger_number_end[3] = trigger_number_end[1]
#                     else: 
#                         if trigger_number_begin[1]==0x7FFFFFFF:
#                             trigger_number_begin[3] = trigger_number_begin[2]
#                             trigger_number_end[3] = trigger_number_end[2]
#                         else:
#                             trigger_number_begin[3] = trigger_number_begin[2]
#                             trigger_number_end[3] = trigger_number_end[1]
#                     #print trigger_number_begin[2],trigger_number_begin[1],"-b-",trigger_number_begin[3]
#                     #print trigger_number_end[2],trigger_number_end[1],"-e-",trigger_number_end[3]
                last_frame_id[plane_id] = frame_id[plane_id]
                # Reset counter
                event_status[plane_id] = 0
                word_index[plane_id] = 0
                timestamp[plane_id] = get_timestamp_low(word)

            elif word_index[plane_id] == -1:  # trash data
                # TODO: add event status trash data
                pass
            else:  # correct M26 data
                word_index[plane_id] += 1
                if word_index[plane_id] == 1:  # 1. timestamp high
                    timestamp[plane_id] = get_timestamp_high(word) | timestamp[plane_id]
                    #event_number[plane_id] = get_event_number(plane_id, event_number, timestamp)

                elif word_index[plane_id] == 2:  # 2. word should have the frame ID high word
                    frame_id[plane_id] = get_frame_id_high(word)

                elif word_index[plane_id] == 3:  # 3. word should have the frame ID low word
                    frame_id[plane_id] = get_frame_id_low(word) | frame_id[plane_id]
                    if plane_id == 0:
                        frame_id[0] = frame_id[plane_id]

                elif word_index[plane_id] == 4:  # 4. word should have the frame length high word
                    frame_length[plane_id] = get_frame_length(word)

                elif word_index[plane_id] == 5:  # 5. word should have the frame length low word (=high word, one data line, the number of words is repeated 2 times)
                    if frame_length[plane_id] != get_frame_length(word):
                        add_event_status(plane_id, event_status, EVENT_INCOMPLETE)

                elif word_index[plane_id] == 6 + frame_length[plane_id]:  # Second last word is frame tailer high word
                    if not is_frame_tailer_high(word):
                        add_event_status(plane_id, event_status, DATA_ERROR)

                elif word_index[plane_id] == 7 + frame_length[plane_id]:  # First last word is frame tailer low word
                    frame_length[plane_id] = -1
                    n_words[plane_id] = 0
                    word_index[plane_id] = -1
                    if not is_frame_tailer_low(word, plane=plane_id + 1):
                        add_event_status(plane_id, event_status, DATA_ERROR)

                else:  # Column / Row words (actual data word)
                    if n_words[plane_id] == 0:  # First word containing the row info and the number of data words for this row
                        if word_index[plane_id] == 6 + frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                            add_event_status(plane_id, event_status, UNEVEN_EVENT)
                        else:
                            n_words[plane_id] = get_n_words(word)
                            row[plane_id] = get_row(word)
                        if has_overflow(word):
                            add_event_status(plane_id, event_status, DATA_ERROR)
                            n_words[plane_id] = 0
                        if row[plane_id] > 576:  # row overflow
                            add_event_status(plane_id, event_status, ROW_OVERFLOW)
                    else:
                        n_words[plane_id] = n_words[plane_id] - 1  # Count down the words
                        n_hits = get_n_hits(word)
                        column = get_column(word)
                        if column >= 1152:  # column overflow
                            add_event_status(plane_id, event_status, COL_OVERFLOW)
                        for k in range(n_hits + 1):
                            out_trigger_number = 0 if trigger_number < 0 else trigger_number  # Prevent storing negative number in unsigned int
                            if hit_buffer_index[plane_id] < max_hits_per_event:
                                # store hits in hit buffer array
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['event_number'] = event_number[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['trigger_number'] = out_trigger_number
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['timestamp'] = timestamp[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['plane'] = plane_id +1
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['frame'] = frame_id[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['column'] = column + k
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['row'] = row[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['trigger_number_end'] = trigger_number_end[3]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['trigger_number_begin'] = trigger_number_begin[3]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['event_status'] = event_status[plane_id]
                                hit_buffer_index[plane_id] += 1
                            else:
                                # truncated data
                                add_event_status(plane_id, event_status, TRUNC_EVENT)
        elif is_trigger_word(word):  # raw data word is TLU word
            trigger_number_end[0] = get_trigger_number(word, trigger_data_format)
            trigger_number = get_trigger_number(word, trigger_data_format)
            if trigger_number_begin[0] == 0x7FFFFFFF:
                trigger_number_begin[0] = trigger_number_end[0]
#             if True: #(debug & 1) == 1:
#                 hits[hit_index]['event_number'] = event_number[0]
#                 hits[hit_index]['trigger_number_begin'] = trigger_number_begin[0]
#                 hits[hit_index]['timestamp'] = get_trigger_timestamp(word)
#                 hits[hit_index]['plane'] = 0
#                 hits[hit_index]['frame'] = frame_id[0]
#                 hits[hit_index]['column'] = 0
#                 hits[hit_index]['trigger_number_end'] = trigger_number_end[0]
#                 hits[hit_index]['trigger_status'] = 1
#                 hits[hit_index]['event_status'] = 0
#                 hits[hit_index]['row'] = 0
#                 hit_index=hit_index+1
#             ### trigger_timestamp = get_trigger_timestamp(word)  ## TODO compare with timestamp of m26

    return (hits[:hit_index], frame_id, last_frame_id, frame_length, word_index, n_words, row, hits_buffer, hit_buffer_index,
            event_status, event_number, trigger_number, trigger_number_begin, trigger_number_end, timestamp)


class RawDataInterpreter(object):
    ''' Class to convert the raw data chunks to hits'''

    def __init__(self, max_hits_per_event=1000, debug=1, trigger_data_format=0):
        self.max_hits_per_event = max_hits_per_event
        self.debug = debug
        self.trigger_data_format = trigger_data_format
        self.reset()

    def reset(self):  # Reset variables
        # Per frame variables
        self.frame_id = np.zeros(6, np.int32)  # The counter value of the actual frame
        self.last_frame_id = np.ones(6, np.int32) * -1  # The counter value of the last frame
        self.frame_length = np.ones(6, np.int32) * -1  # The number of data words in the actual frame
        self.word_index = np.ones(6, np.int32) * -1  # The word index per device of the actual frame
        self.timestamp = np.ones(6, np.int32) * -1  # The word index per device of the actual frame
        self.n_words = np.zeros(6, np.uint32)  # The number of words containing column / row info
        self.row = np.ones(6, np.uint32) * 0xffffffff  # the actual readout row (rolling shutter)

        # Per event variables
        self.hits_buffer = np.zeros((6, self.max_hits_per_event), dtype=hit_dtype)  # Buffers actual event hits, needed since raw data is analyzed in chunks
        self.hit_buffer_index = np.zeros(6, np.uint32)  # Hit buffer index for each plane; needed to append hits
        self.event_status = np.zeros(shape=(6, ), dtype=np.uint16)  # Actual event status for each plane
        self.event_number = np.ones(6, np.int64) * -1  # The event counter set by the software counting full events for each plane
        self.trigger_number = -1  # The actual event trigger number
        self.trigger_number_begin = np.ones(4, np.uint16) * 0xFFFF   # The event trigger number begin
        self.trigger_number_end = np.ones(4, np.uint16) * 0xFFFF   # The event trigger number end

    def interpret_raw_data(self, raw_data):
        chunk_result = build_hits(raw_data=raw_data,
                                  frame_id=self.frame_id,
                                  last_frame_id=self.last_frame_id,
                                  frame_length=self.frame_length,
                                  word_index=self.word_index,
                                  n_words=self.n_words,
                                  row=self.row,
                                  hits_buffer=self.hits_buffer,
                                  hit_buffer_index=self.hit_buffer_index,
                                  event_status=self.event_status,
                                  event_number=self.event_number,
                                  trigger_number=self.trigger_number,
                                  trigger_number_begin=self.trigger_number_begin,
                                  trigger_number_end=self.trigger_number_end,
                                  timestamp=self.timestamp,
                                  max_hits_per_event=self.max_hits_per_event,
                                  debug=self.debug,
                                  trigger_data_format=self.trigger_data_format)

        # Set updated buffer variables
        (hits,
         self.frame_id,
         self.last_frame_id,
         self.frame_length,
         self.word_index,
         self.n_words,
         self.row,
         self.hits_buffer,
         self.hit_buffer_index,
         self.event_status,
         self.event_number,
         self.trigger_number,
         self.trigger_number_being,
         self.trigger_number_end,
         self.timestamp) = chunk_result
         

        return hits
