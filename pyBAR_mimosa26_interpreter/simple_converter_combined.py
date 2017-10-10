#!/usr/bin/env python
''' Intepretation of telescope data from Mimosa and FE-I4. Uses Mimosa time stamp
somehow. For trigger mode 2.
'''

import sys
import time
import os
from numba import njit
import numpy as np
import tables

hit_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'),
                      ('trigger_time_stamp', '<u4'),
                      ('x', '<u2'), ('y', '<u2'), ('frame', '<u4')])
hit_buf_dtype = np.dtype([('frame', '<u4'), ('x', '<u2'), ('y', '<u2')])
tlu_buf_dtype = np.dtype([('event_number', '<i8'), ('trigger_number', '<u4'), ('trigger_time_stamp', '<u4'), ('frame', '<u4')])


@njit
def _m26_converter(raw_data, plane, hits, mframe, dlen, idx, numstatus, row, ovf,
                   tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug):
    hit_i = 0
    jj = 0

    for raw_i in range(raw_data.shape[0]):
        raw_d = raw_data[raw_i]
        if (0xFFF00000 & raw_d) == (0x20000000 | (plane << 20)):  # M26
            # print raw_i,hex(raw_d),idx
            if (0x00020000 & raw_d == 0x20000):
                idx = -1
                # print raw_i,hex(raw_d),mid,idx[mid],"reset frame data because of data loss"
            elif (0x000F0000 & raw_d == 0x10000):
                # timestamp[plane] = raw_d & 0xFFFF
                idx = 0
                # print raw_i,hex(raw_d),"frame start"
            elif idx == -1:
                # print raw_i,hex(raw_d),mid,idx[mid],"trash"
                pass
            else:
                idx = idx + 1
                if idx == 1:
                    pass
                    # timestamp = (0x0000FFFF & raw_d) << 16 |timestamp
                    # print raw_i,hex(raw_d),mid,idx[mid],"timestamp", timestamp[plane]
                elif idx == 2:
                    mframe = (0x0000FFFF & raw_d)
                elif idx == 3:
                    mframe = (0x0000FFFF & raw_d) << 16 | mframe
                    # print raw_i,hex(raw_d),idx,"mframe", mframe
                elif idx == 4:
                    dlen = (raw_d & 0x0000FFFF) * 2
                    # print raw_i,hex(raw_d),mid,idx[mid],"dlen", dlen[mid]
                elif idx == 5:
                    if dlen != (raw_d & 0x0000FFFF) * 2:
                        return hit_i, raw_i, 3, mframe, dlen, idx, numstatus, row, ovf, \
                               tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
                elif idx == 6 + dlen:
                    if raw_d & 0xFFFF != 0xaa50:
                        return hit_i, raw_i, 4, mframe, dlen, idx, numstatus, row, ovf, \
                               tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
                elif idx == 7 + dlen:  # Last word is frame tailer low word
                    dlen = -1
                    numstatus = 0
                    if raw_d & 0xFFFF != (0xaa50 | plane):
                        return hit_i, raw_i, 5, mframe, dlen, idx, numstatus, row, ovf, \
                               tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
                    # copy to hits
                    jj = 0
                    for j in range(tlu_buf_i):
                        if tlu_buf[j]["frame"] == mframe - 2:
                            for i in range(hit_buf_i):
                                if hit_buf[i]['frame'] == mframe - 1 or hit_buf[i]['frame'] == mframe:
                                    hits[hit_i]["trigger_number"] = tlu_buf[j]["trigger_number"]
                                    hits[hit_i]["trigger_time_stamp"] = tlu_buf[j]["trigger_time_stamp"]
                                    hits[hit_i]["event_number"] = tlu_buf[j]["event_number"]
                                    hits[hit_i]['x'] = hit_buf[i]['x']
                                    hits[hit_i]['y'] = hit_buf[i]['y']
                                    hits[hit_i]['frame'] = hit_buf[i]['frame']
                                    hit_i = hit_i + 1
                                # else :#do nothing
                        elif tlu_buf[j]['frame'] == mframe - 1 or tlu_buf[j]['frame'] == mframe:
                            tlu_buf[jj]["trigger_number"] = tlu_buf[j]["trigger_number"]
                            tlu_buf[jj]["trigger_time_stamp"] = tlu_buf[j]["trigger_time_stamp"]
                            tlu_buf[jj]["frame"] = tlu_buf[j]["frame"]
                            tlu_buf[jj]["event_number"] = tlu_buf[j]["event_number"]
                            jj = jj + 1
                    tlu_buf_i = jj
                    jj = 0
                    for i in range(hit_buf_i):
                        if hit_buf[i]['frame'] == mframe:
                            hit_buf[jj]['frame'] = hit_buf[i]['frame']
                            hit_buf[jj]['y'] = hit_buf[i]['y']
                            hit_buf[jj]['x'] = hit_buf[i]['x']
                            hit_buf[jj]['frame'] = hit_buf[i]['frame']
                            jj = jj + 1
                    hit_buf_i = jj
                    if hit_i > hits.shape[0] - 1000:
                        break
                else:
                    if numstatus == 0:
                        if idx == 6 + dlen - 1:
                            pass
                        else:
                            numstatus = (raw_d) & 0xF
                            row = (raw_d >> 4) & 0x7FF
                        if raw_d & 0x8000 == 0x8000:
                            ovf = ovf + 1
                            numstatus = 0
                            return hit_i, raw_i, 8, mframe, dlen, idx, numstatus, row, ovf, \
                                tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug

                        if row > 576:
                            return hit_i, raw_i, 1, mframe, dlen, idx, numstatus, row, ovf, \
                                tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
                    else:
                        numstatus = numstatus - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if col == 0x5C0:
                            return hit_i, raw_i, 9, mframe, dlen, idx, numstatus, row, ovf, \
                                tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
            # MIMOSA_COL_OVF?_WARN
                        elif col >= 1152:
                            return hit_i, raw_i, 2, mframe, dlen, idx, numstatus, row, ovf, \
                                tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
            # #MIMOSA_COL_ERROR

                        for k in range(num + 1):
                            hit_buf[hit_buf_i]['frame'] = mframe
                            hit_buf[hit_buf_i]['x'] = col + k
                            hit_buf[hit_buf_i]['y'] = row
                            hit_buf_i = hit_buf_i + 1
        elif(0x80000000 & raw_d == 0x80000000):  # TLU
            tlu_buf[tlu_buf_i]["trigger_number"] = raw_d & 0x0000FFFF
            tlu_buf[tlu_buf_i]["trigger_time_stamp"] = (raw_d & 0x7FFF0000) >> 16
            tlu_buf[tlu_buf_i]["frame"] = mframe
            tlu_buf[tlu_buf_i]["event_number"] = event_number
            # rint raw_i,hex(raw_d),"tlu",mframe, raw_d & 0x0000FFFF ,event_number
            tlu_buf_i = tlu_buf_i + 1
            if tlu_buf_i == tlu_buf.shape[0]:
                return hit_i, raw_i, 10, mframe, dlen, idx, numstatus, row, ovf, \
                    tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug
  # #TLU_BUF_OVF_WANNING
            event_number = event_number + 1

    return hit_i, raw_i, 0, mframe, dlen, idx, numstatus, row, ovf, \
        tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug


def m26_converter(fin, fout, plane):
    start = 0
    n = 10000000

    mframe = 0
    dlen = -1
    idx = -1
    numstatus = 0
    row = 0
    event_status = 0
    event_number = np.uint64(0)
    hits = np.empty(n * 10, dtype=hit_dtype)
    tlu_buf = np.empty(10240000, dtype=tlu_buf_dtype)  # changed the vlaue, before 1024
    hit_buf = np.empty(40960000, dtype=hit_buf_dtype)  # changed the value, before 4096
    ovf = 0
    tlu_buf_i = 0
    hit_buf_i = 0
    debug = 1

    with tables.open_file(fin) as tb:
        end = int(len(tb.root.raw_data))
        print "fout:", fout, "number of data:", end
        t0 = time.time()
        hit = np.empty(2 * n, dtype=hit_dtype)

        with tables.open_file(fout, 'w') as out_file_h5:
            description = np.zeros((1,), dtype=hit_dtype).dtype
            hit_table = out_file_h5.create_table(out_file_h5.root,
                                                 name='Hits',
                                                 description=description,
                                                 title='hit_data')
            while True:
                tmpend = min(start + n, end)
                hit_i, raw_i, err, mframe, dlen, idx, numstatus, row, ovf, \
                    tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug = \
                    _m26_converter(tb.root.raw_data[start:tmpend], plane, hits, mframe, dlen, idx, numstatus, row, ovf,
                                   tlu_buf, tlu_buf_i, hit_buf, hit_buf_i, event_number, debug)
                t1 = time.time() - t0
                if err == 0:
                    print start, raw_i, hit_i, err, "---%.3f%% %.3fs(%.3fus/dat)" % ((tmpend * 100.0) / end, t1, (t1) / tmpend * 1.0E6)
                else:  # # Fix error code
                    if err == 1:
                        err_str = "MIMOSA_ROW_ERROR",
                    elif err == 2:
                        err_str = "MIMOSA_COL_ERROR",
                    elif err == 3:
                        err_str = "MIMOSA_DLEN_ERROR",
                    elif err == 4:
                        err_str = "MIMOSA_TAILER_ERROR",
                    elif err == 5:
                        err_str = "MIMOSA_TAILER2_ERROR",
                    elif err == 6:
                        err_str = "FEI4_TOT1_ERROR",
                    elif err == 7:
                        err_str = "FEI4_TOT2_ERROR",
                    elif err == 8:
                        err_str = "MIMOSA_OVF_WARN",
                    elif err == 9:
                        err_str = "MIMOSA_COL_OVF?_WARN",
                    elif err == 10:
                        err_str = "TLU_BUF_OVF_WANNING",
                        tlu_buf_i = 125
                        tlu_buf[:125] = tlu_buf[-125:]
                    print err_str, start, raw_i, hex(tb.root.raw_data[start + raw_i])
                    # for j in range(-100,100,1):
                    #    print "ERROR %4d"%j,start+raw_i+j,hex(tb.root.raw_data[start+raw_i+j])
                    # break
                hit_table.append(hits[:hit_i])
                hit_table.flush()
                start = start + raw_i + 1
                if start >= end:
                    break


aligned_dtype = np.dtype([('event_number', '<i8'), ('frame', 'u1'),
                          ('column', '<u2'), ('row', '<u2'), ('charge', '<u2')])
aligned_dtype2 = np.dtype([('event_number', '<i8'), ('frame', 'u4'),
                           ('column', '<u2'), ('row', '<u2'), ('charge', '<u2')])


@njit
def _align_event_number(fe_hits, m26_hits, hits, tr, frame, debug):
    hit_i = 0
    m26_i = 0
    for fe_i in range(fe_hits.shape[0]):
        fe_trig = fe_hits[fe_i]["trigger_number"]
        fe_ts = fe_hits[fe_i]["trigger_time_stamp"]
        while m26_i < m26_hits.shape[0]:
            # print fe_i, m26_i, fe_hits[fe_i]["event_number"], m26_hits[m26_i]["event_number"],
            # print hex(fe_hits[fe_i]["trigger_number"]),hex(m26_hits[m26_i]["trigger_number"]),
            m26_trig = m26_hits[m26_i]["trigger_number"]
            m26_ts = m26_hits[m26_i]["trigger_time_stamp"]
            if (m26_trig - fe_trig) & 0x4000 == 0x4000:
                # print 'increase only m26_i'
                m26_i = m26_i + 1
            # check for timestamp + trigger_number, need to check both because can have TLU overflow, which fakes trigger number
            elif m26_trig == fe_trig and m26_ts == fe_ts:
                hits[hit_i]["event_number"] = fe_hits[fe_i]["event_number"]
                if tr is True:
                    hits[hit_i]["row"] = m26_hits[m26_i]["x"] + 1
                    hits[hit_i]["column"] = m26_hits[m26_i]["y"] + 1
                else:
                    hits[hit_i]["row"] = m26_hits[m26_i]["y"] + 1
                    hits[hit_i]["column"] = m26_hits[m26_i]["x"] + 1
                if frame is True:
                    hits[hit_i]['frame'] = m26_hits[m26_i]['frame']
                else:
                    hits[hit_i]['frame'] = 0
                hits[hit_i]['charge'] = 1
                hit_i = hit_i + 1
                m26_i = m26_i + 1
                # print 'matched'
            # check cas that TLU fakes trigger_number because of overflow
            elif m26_trig == fe_trig and m26_ts != fe_ts:
                # print 'return'
                return fe_i, m26_i, hit_i, 4
            else:  # m26_trig > fe_trig
                # print 'break'
                break
        if m26_i == m26_hits.shape[0]:
            if hit_i == 0:
                # print "ev",fe_hits[0]["event_number"],m26_hits[0]["event_number"]
                # print "trig",hex(fe_hits[0]["trigger_number"]),hex(m26_hits[0]["trigger_number"])
                return fe_i, m26_i, hit_i, 1
            return fe_i, m26_i, hit_i, 0
    return fe_i + 1, m26_i, hit_i, 3


def align_event_number(fin, fe_fin, fout, tr=False, frame=False):
    m26_start = 0
    fe_start = 0
    n = 10000000

    debug = 1
    print fout
    m26_tb = tables.open_file(fin)
    m26_end = int(len(m26_tb.root.Hits))
    print "m26", m26_end,
    fe_tb = tables.open_file(fe_fin)
    fe_end = int(len(fe_tb.root.Hits))
    print "fe", fe_end
    with tables.open_file(fout, 'w') as out_tb:
        if frame is True:
            description = np.zeros((1,), dtype=aligned_dtype2).dtype
        else:
            description = np.zeros((1,), dtype=aligned_dtype).dtype
        hit_table = out_tb.create_table(out_tb.root,
                                        name='Hits',
                                        description=description,
                                        title='hit_data')
        t0 = time.time()
        if frame is True:
            hits = np.empty(n, dtype=aligned_dtype2)
        else:
            hits = np.empty(n, dtype=aligned_dtype)

        while True:
            fe_tmpend = min(fe_start + n / 10, fe_end)
            m26_tmpend = min(m26_start + n, m26_end)
            fe_i, m26_i, hit_i, err = _align_event_number(fe_tb.root.Hits[fe_start:fe_tmpend],
                                                          m26_tb.root.Hits[m26_start:m26_tmpend],
                                                          hits, tr, frame, debug)
            t1 = time.time() - t0
            if err == 0:
                print fe_start, m26_start, fe_i, m26_i, hit_i, err, "---%.3f%% %.3fs(%.3fus/dat)" % ((m26_tmpend * 100.0) / m26_end, t1, (t1) / m26_tmpend * 1.0E6)
            else:
                # print "ERROR=%d" % err, hit_i, fe_i, hex(fe_tb.root.Hits[fe_start + fe_i]["event_number"]),
                # print m26_i, hex(m26_tb.root.Hits[m26_start + hit_i]["event_number"])
                m26_i = m26_i + 1
                fe_i = fe_i + 1
            hit_table.append(hits[:hit_i])
            hit_table.flush()
            fe_start = fe_start + fe_i
            m26_start = m26_start + m26_i
            if fe_start >= fe_end or m26_start >= m26_end:
                break
    m26_tb.close()
    fe_tb.close()


if __name__ == "__main__":
    debug = 1
    tr = False
    if len(sys.argv) < 2:
        print "simple_converter.py [-tr] <input file>"
    fin = sys.argv[-1]
    if "-tr" in sys.argv:
        tr = True

    for i in range(1, 7):
        ftmp = os.path.join(fin[:-3] + "_event_aligned%d.h5" % i)
        m26_converter(fin, ftmp, i)

        fout = os.path.join(fin[:-3] + "_aligned%d.h5" % i)
        fe_fin = os.path.join(fin[:-3] + "_event_aligned.h5")
        align_event_number(ftmp, fe_fin, fout, tr=tr)
        if debug == 0:
            os.remove(ftmp)
    if debug == 0:
        os.remove(fe_fin)