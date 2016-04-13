''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''
from numba import njit
import numpy as np


hit_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('plane', '<u2'), ('frame', '<u4'), ('column', '<u2'), ('row', '<u4'), ('trigger_status', '<u1'), ('event_status', '<u2')])


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
NO_HIT = 2048  # vents without any hit, usefull for trigger number debugging


# Old interpretation code from Toko Hirono, kept for now for reference
@njit
def _m26_decode_orig(raw, dat, idx,mframe,timestamp,dlen,numstatus,row,tlu,lv1,ovf): 
    fetot = 0
    fecol = 0
    ferow = 0
    felv1=0
    raw_i = 0
    raw_d = 0
    hit = 0
    mid = 0   ## id for mimosa plane 0,1,2,3,4,5
    plane = 0 ## plane number 0:FEI3, 1-6:mimosa, F:TLU
    
    end = len(raw)
    
    while raw_i < end:
        raw_d = raw[raw_i]
        if (0xFF000000 & raw_d == 0x20000000): #M26
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x00020000 & raw_d == 0x20000):
                idx[0] = -1
                idx[1] = -1
                idx[2] = -1
                idx[3] = -1
                idx[4] = -1
                idx[5] = -1
            elif (0x000F0000 & raw_d == 0x10000):
                timestamp[plane] = raw_d & 0xFFFF
                idx[mid] = 0
            elif idx[mid] == -1:
                pass
            else:
                idx[mid] = idx[mid] + 1
                if idx[mid] == 1:
                    timestamp[plane] = (0x0000FFFF & raw_d) << 16 |timestamp[plane]
                elif idx[mid] == 2:
                    mframe[mid + 1] = (0x0000FFFF & raw_d)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 | mframe[plane]
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0x0000FFFF) * 2
                elif idx[mid] == 5:
                    if dlen[mid]!=(raw_d & 0x0000FFFF) * 2:
                        return dat[:hit],raw_i,3 ##MIMOSA_DLEN_ERROR
                elif idx[mid] == 6 + dlen[mid]:
                    if raw_d & 0xFFFF != 0xaa50: 
                        return dat[:hit],raw_i,4 ##MIMOSA_TAILER_ERROR
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    if raw_d & 0xFFFF != (0xaa50 | plane): 
                        return dat[:hit],raw_i,5  ##MIMOSA_TAILER2_ERROR
                else:
                    if numstatus[mid] == 0:
                        if idx[mid] == 6 + dlen[mid] - 1:
                            pass
                        else:
                            numstatus[mid] = (raw_d) & 0xF
                            row[mid] = (raw_d >> 4) & 0x7FF
                        if raw_d & 0x8000==0x8000:
                            ovf[mid]=ovf[mid]+1
                            numstatus[mid]==0
                            return dat[:hit],raw_i,8
                        if row[mid]>576:
                            return dat[:hit],raw_i,1
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col>=1152:
                            return dat[:hit],raw_i,2
                        for k in range(num + 1):
                            dat[hit].plane = plane
                            dat[hit].mframe = mframe[plane]
                            dat[hit].timestamp = timestamp[plane]
                            dat[hit].tlu = 0
                            dat[hit].x = col + k
                            dat[hit].y = row[mid]
                            dat[hit].val = 0
                            dat[hit].val2 = 0
                            hit = hit + 1
        elif(0x80000000 & raw_d == 0x80000000): #TLU
            tlu = raw_d & 0xFFFF
            timestamp[0] = (raw_d >>16) & 0x7FFF | (timestamp[1] & 0xFFFF8000) # TODO be more precise.
            if timestamp[0] < timestamp[1]:
                if timestamp[0] < 0xFFFF8000:
                  timestamp[0]= timestamp[0] + 0x8000
                else:
                    timestamp[0]= (timestamp[0]-0xFFFF8000) + 0x8000
            mframe[0] = mframe[1]
            felv1=0
            tm = (raw_d >>16) & 0x7FFF
            dat[hit].plane = -1
            dat[hit].mframe = -1
            dat[hit].timestamp = tm
            dat[hit].tlu = tlu
            dat[hit].x = 0
            dat[hit].y = 0
            dat[hit].val = 0
            dat[hit].val2 = 0
            hit = hit + 1
                            
        elif(0xFF000000 & raw_d == 0x01000000): #FEI4
            if(0xFF0000 & raw_d == 0x00EA0000) | (0xFF0000 & raw_d == 0x00EF0000) |(0xFF0000 & raw_d == 0x00EC0000): ## other data
                pass
            elif (0xFF0000 & raw_d == 0x00E90000): ##BC
                felv1=felv1+1
                ## TODO get lv1, bc
            else: ##TOT1 and TOT2
                fetot=(raw_d & 0x000000F0) >> 4
                fecol=(raw_d & 0x00FE0000) >> 17
                ferow=(raw_d & 0x0001FF00) >> 8
                if fetot !=0xF and fecol<=80 and fecol>=1 and ferow<=336 and ferow>=1:
                    dat[hit].plane = 0
                    dat[hit].mframe = mframe[0]
                    dat[hit].timestamp = timestamp[0]
                    dat[hit].tlu = tlu
                    dat[hit].x = fecol
                    dat[hit].y = ferow
                    dat[hit].val = fetot
                    dat[hit].val2 = felv1
                    hit=hit+1
                else:
                    return dat[:hit],raw_i,6 ## FEI4_TOT1_ERROR
                fetot=(raw_d & 0xF)
                ferow=ferow+1
                if fetot!=0xF:
                    if fecol<=80 and fecol>=1 and ferow<=336 and ferow >=1:
                        #dat[hit] = (0,mframe[0],timestamp[0],tlu, fecol, ferow, tot,lv)
                        dat[hit].plane = 0
                        dat[hit].mframe = mframe[0]
                        dat[hit].timestamp = timestamp[0]
                        dat[hit].tlu = tlu
                        dat[hit].x = fecol
                        dat[hit].y = ferow
                        dat[hit].val = fetot
                        dat[hit].val2 = felv1
                        hit=hit+1
                    else:
                        return dat[:hit],raw_i,7 ##FEI4_TOT2_ERROR
        raw_i = raw_i + 1
        
    return dat[:hit],raw_i,0
    
def m26_decode_orig(fin,fout):
    import tables
    import time
    hit_dtype = np.dtype([('plane', '<u1'),('mframe', '<u4'),('timestamp','<u4'),('tlu', '<u2'),('x', '<u2'), ('y', '<u2'), ('val','<u2'),('val2','<u2')])
    
    mframe = [0] * 7
    timestamp = np.zeros(7,dtype=np.uint32)
    dlen = [-1] * 6
    idx = [-1] * 6
    numstatus = [0] * 6
    ovf = [0] * 6
    row = [-1] * 6
    felv1=-1
    tlu = 0
    start=0
    n = 10000000

    with tables.open_file(fin) as tb:
        end=int(len(tb.root.raw_data))
        print "output",fout,"n of data",end
        t0 = time.time()
        dat = np.empty(n, dtype=hit_dtype)
        dat = dat.view(np.recarray)
        with tables.open_file(fout, 'w') as out_file_h5:
            while True:
                tmpend=min(start+n,end)
                hit_dat,raw_i,err =_m26_decode_orig(tb.root.raw_data[start:tmpend],dat, idx,mframe,timestamp,dlen,numstatus,row,tlu,felv1,ovf)
                t1=time.time()-t0
                if err==0:
                    print start,raw_i,len(hit_dat),ovf,"---%.3f%% %.3fs(%.3fus/dat)"%((tmpend*100.0)/end, t1, (t1)/tmpend*1.0E6)
                else:
                    if err==1:
                        print "MIMOSA_ROW_ERROR",
                    elif err==2:
                        print "MIMOSA_COL_ERROR",
                    elif err==3:
                        print "MIMOSA_DLEN_ERROR",
                    elif err==4:
                        print "MIMOSA_TAILER_ERROR",
                    elif err==5:
                        print "MIMOSA_TAILER2_ERROR",
                    elif err==6:
                        print "FEI4_TOT1_ERROR",
                    elif err==7:
                        print "FEI4_TOT2_ERROR",
                    elif err==8:
                        print "MIMOSA_OVF_WARN",
                    print err,start,raw_i,hex(tb.root.raw_data[start+raw_i])
                    for j in range(-100,100,1):
                        print "ERROR %4d"%j,start+raw_i+j,hex(tb.root.raw_data[start+raw_i+j])
                    break
                if start==0:
                    description = np.zeros((1, ), dtype=hit_dtype).dtype
                    hit_table = out_file_h5.create_table(out_file_h5.root, 
                        name='Hits', 
                        description=description, 
                        title='hit_data')
                hit_table.append(hit_dat)
                hit_table.flush()
                start=start+raw_i
                if start>=end:
                    break
            


@njit
def is_mimosa_data(word):  # Check for Mimosa data word
    return 0xF0000000 & word == 0x20000000


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
def get_frame_length(word):
    return (word & 0xFFFF) * 2


@njit
def get_row(word):
    return (word >> 4) & 0x7FF


@njit
def get_column(word):
    return (word >> 2) & 0x7FF


@njit
def is_frame_header_high(word):  # Check if frame header high word
    return 0x000FFFFF & word == 0x15555


@njit
def is_frame_header_low(word, plane):  # Check if frame header low word for the actual plane
    return (0x0000FFFF & word) == (0x5550 | plane)


@njit
def is_frame_tailer_high(word):  # Check if frame header high word
    return word & 0xFFFF == 0xaa50


@njit
def is_frame_tailer_low(word, plane):  # Check if frame header low word for the actual plane
    return (word & 0xFFFF) == (0xaa50 | plane)


@njit
def get_n_words(word):  # Return the number of data words for the actual row
    return word & 0xF


@njit
def get_n_hits(word):  # Returns the number of hits given by actual column word
    return word & 0x3


@njit
def has_overflow(word):
    return word & 0x00008000 != 0


@njit
def is_trigger_word(word):
    return 0x80000000 & word == 0x80000000


@njit
def get_trigger_number(word):  # Returns the number of hits given by actual column word
    return word & 0xFFFF


@njit
def add_event_status(plane_id, event_status, status_code):
    # print 'ADD EVENT STATUS', status_code
    event_status[plane_id] |= status_code


@njit
def finish_event(plane_id, hits_buffer, hit_buffer_index, event_status, hits, hit_index):  # Append buffered hits to hit object
    for i_hit in range(hit_buffer_index):  # Loop over buffered hits
        hits[hit_index] = hits_buffer[plane_id, i_hit]
        hits[hit_index].event_status = event_status
        hit_index += 1
    return hit_index  # Return actual hit index; needed to append correctly at next call of finish_event


@njit
def build_hits(raw_data, frame_id, last_frame_id, frame_length, word_index, n_words, row, hits_buffer, hit_buffer_index, event_status, event_number, trigger_number, max_hits_per_event):
    ''' Main interpretation function. Loops over the raw data an creates a hit array. Data errors are checked for.
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
    trigger_number : number
        The actual event trigger number
    max_hits_per_event : number
        Maximum expected hits per event. Needed to allocate hit buffer.

    Returns
    -------
    A list of all inpur parameters, but raw_data is exchanged for a hit array and max_hits_per_event is not returned.
    '''
    # The raw data order of the Mimosa 26 data should be always START / FRAMEs ID / FRAME LENGTH / DATA
    # Since the clock is the same for each plane; the order is START plane 1, START plane 2, ...

    hits = np.zeros(shape=(raw_data.shape[0]), dtype=hit_dtype)  # Result hits array
    hit_index = 0  # Pointer to actual hit in resul hit arrray; needed to append hits every event

    for raw_i in range(raw_data.shape[0]):
        word = raw_data[raw_i]  # Actual raw data word
        if is_mimosa_data(word):  # There can be not mimosa related data words (from FE-I4)

            # Check to which plane the data belongs
            actual_plane = get_plane_number(word)
            plane_id = actual_plane - 1  # The actual_plane if the actual word belongs to (0 .. 5)

            # Interpret the word of the actual plane
            if is_frame_header_high(word):  # New event for actual plane; events are aligned at this header
                # Finish old event
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
            else:
                word_index[plane_id] += 1
                if word_index[plane_id] == 1:  # 1. word should have the header low word
                    if not is_frame_header_low(word, actual_plane):
                        add_event_status(plane_id, event_status, DATA_ERROR)
                elif word_index[plane_id] == 2:  # 2. word should have the frame ID high word
                    frame_id[plane_id + 1] = get_frame_id_high(word)
                elif word_index[plane_id] == 3:  # 3. word should have the frame ID low word
                    frame_id[actual_plane] = get_frame_id_low(word) | frame_id[actual_plane]
                    if plane_id == 0:
                        frame_id[0] = frame_id[actual_plane]
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
                    if not is_frame_tailer_low(word, actual_plane):
                        add_event_status(plane_id, event_status, DATA_ERROR)
                else:  # Column / Row words
                    if n_words[plane_id] == 0:  # First word containing the row info and the number of data words for this row
                        if word_index[plane_id] == 6 + frame_length[plane_id] - 1:  # Always even amount of words or this fill word is used
                            add_event_status(plane_id, event_status, UNEVEN_EVENT)
                        else:
                            n_words[plane_id] = get_n_words(word)
                            row[plane_id] = get_row(word)
                            if has_overflow(word):
                                add_event_status(plane_id, event_status, DATA_ERROR)
                    else:
                        if trigger_number < 0:  # Trigger number < 0 means no trigger number
                            add_event_status(plane_id, event_status, NO_TRG_WORD)
                        n_words[plane_id] = n_words[plane_id] - 1  # Count down the words
                        n_hits = get_n_hits(word)
                        column = get_column(word)
                        for k in range(n_hits + 1):
                            out_trigger_number = 0 if trigger_number < 0 else trigger_number  # Prevent storing negative number in unsigned int
                            if hit_buffer_index[plane_id] < max_hits_per_event:
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['event_number'] = event_number[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['trigger_number'] = out_trigger_number
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['plane'] = plane_id
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['frame'] = frame_id[plane_id]
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['column'] = column + k
                                hits_buffer[plane_id, hit_buffer_index[plane_id]]['row'] = row[plane_id]
                                hit_buffer_index[plane_id] += 1
                            else:
                                add_event_status(plane_id, event_status, TRUNC_EVENT)
        elif is_trigger_word(word):
            trigger_number = get_trigger_number(word)

    return (hits[:hit_index], frame_id, last_frame_id, frame_length, word_index, n_words, row, hits_buffer, hit_buffer_index, event_status, event_number, trigger_number)


class RawDataInterpreter(object):
    ''' Class to convert the raw data chunks to hits'''

    def __init__(self, max_hits_per_event=1000, debug=False):
        self.max_hits_per_event = max_hits_per_event
        self.debug = debug
        self.reset()

    def reset(self):  # Reset variables
        # Per frame variables
        self.frame_id = [0] * 8  # The counter value of the actual frame
        self.last_frame_id = [-1] * 8  # The counter value of the last frame
        self.frame_length = [-1] * 6  # The number of data words in the actual frame
        self.word_index = [-1] * 6  # The word index per device of the actual frame
        self.n_words = [0] * 6  # The number of words containing column / row info
        self.row = [-1] * 6  # the actual readout row (rolling shutter)

        # Per event variables
        self.hits_buffer = np.zeros((6, self.max_hits_per_event), dtype=hit_dtype)  # Buffers actual event hits, needed since raw data is analyzed in chunks
        self.hit_buffer_index = [0] * 6  # Hit buffer index for each plane; needed to append hits
        self.event_status = np.zeros(shape=(6, ), dtype=np.uint16)  # Actual event status for each plane
        self.event_number = [-1] * 8  # The event counter set by the software counting full events for each plane
        self.trigger_number = -1  # The actual event trigger number

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
                                  max_hits_per_event=self.max_hits_per_event)

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
         self.trigger_number) = chunk_result

        return hits
