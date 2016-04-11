''' Script to check the correctness of the interpretation.
'''

import unittest
import tables as tb
import numpy as np

from pyBAR_mimosa26_interpreter import data_interpreter


def m26_decode_orig(raw, start=0, end=-1):  # Old quick and dirty Mimosa26 interpretation from Toko Hirono; used for comparison
    debug = 0
    n = 10000
    mframe = [0] * 8
    dlen = [-1] * 6
    idx = [-1] * 6
    numstatus = [0] * 6
    row = [-1] * 6
    dat = np.empty(n, dtype=[('plane', '<u2'), ('mframe', '<u4'), ('x', '<u2'), ('y', '<u2'), ('tlu', '<u2')])
    raw_i = start
    if end > 0:
        end = min(len(raw), end)
    else:
        end = len(raw)
    hit = 0
    while raw_i < end:
        raw_d = raw[raw_i]
        if hit + 4 >= n:
            hit = 0
        if (0xF0000000 & raw_d == 0x20000000):
            if debug:
                print raw_i, hex(raw_d),
            plane = ((raw_d >> 20) & 0xF)
            mid = plane - 1
            if (0x000FFFFF & raw_d == 0x15555):
                idx[mid] = 0
            elif idx[mid] == -1:
                if debug:
                    print "trash"
            else:
                idx[mid] = idx[mid] + 1
                if debug:
                    print mid, idx[mid],
                if idx[mid] == 1:
                    if debug:
                        print "header"
                    if (0x0000FFFF & raw_d) != (0x5550 | plane):
                        print "header ERROR", hex(raw_d)
                elif idx[mid] == 2:
                    if debug:
                        print "frame lsb"
                    mframe[mid + 1] = (0x0000FFFF & raw_d)
                elif idx[mid] == 3:
                    mframe[plane] = (0x0000FFFF & raw_d) << 16 | mframe[plane]
                    if mid == 0:
                        mframe[0] = mframe[plane]
                    if debug:
                        print "frame", mframe[plane]
                elif idx[mid] == 4:
                    dlen[mid] = (raw_d & 0xFFFF) * 2
                    if debug:
                        print "length", dlen[mid]
                elif idx[mid] == 5:
                    if debug:
                        print "length check"
                    if dlen[mid] != (raw_d & 0xFFFF) * 2:
                        print "dlen ERROR", hex(raw_d)
                elif idx[mid] == 6 + dlen[mid]:
                    if debug:
                        print "tailer"
                    if raw_d & 0xFFFF != 0xaa50:
                        print "tailer ERROR", hex(raw_d)
                elif idx[mid] == 7 + dlen[mid]:
                    dlen[mid] = -1
                    numstatus[mid] = 0
                    if debug:
                        print "frame end"
                    if (raw_d & 0xFFFF) != (0xaa50 | plane):
                        print "tailer ERROR", hex(raw_d)
                else:
                    if numstatus[mid] == 0:
                        if idx[mid] == 6 + dlen[mid] - 1:
                            if debug:
                                print "pass"
                            pass
                        else:
                            numstatus[mid] = (raw_d) & 0xF
                            row[mid] = (raw_d >> 4) & 0x7FF
                            if debug:
                                print "sts", numstatus[mid], "row", row[mid]
                            if raw_d & 0x00008000 != 0:
                                print "overflow", hex(raw_d)
                                break
                    else:
                        numstatus[mid] = numstatus[mid] - 1
                        num = (raw_d) & 0x3
                        col = (raw_d >> 2) & 0x7FF
                        if debug:
                            print "col", col, "num", num
                        for k in range(num + 1):
                            dat[hit] = (plane, mframe[plane], col + k, row[mid], 0)
                            hit = hit + 1
        elif(0x80000000 & raw_d == 0x80000000):
            tlu = raw_d & 0xFFFF
            if debug:
                print hex(raw_d)
            dat[hit] = (7, mframe[0], 0, 0, tlu)
            hit = hit + 1
        raw_i = raw_i + 1
    if debug:
        print "raw_i", raw_i

    return dat


class TestInterpretation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_interpretation_function(self):
        with tb.open_file('example_data_1.h5', 'r') as in_file_h5:
            raw_data = in_file_h5.root.raw_data[:]

            hits = data_interpreter.build_hits(raw_data)
            hits_old = m26_decode_orig(raw_data)

            # Check interpretation
            for plane in range(1, 7):  # Hits are sorted differently per plane, thus loop is needed
                new = hits[hits['plane'] + 1 == plane]['column']
                old = hits_old[np.where(hits_old['plane'] == plane)]['x'][:new.shape[0]]
                np.testing.assert_array_equal(old, new, err_msg='plane %d' % (plane - 1))

                new = hits[hits['plane'] + 1 == plane]['row']
                old = hits_old[np.where(hits_old['plane'] == plane)]['y'][:new.shape[0]]
                np.testing.assert_array_equal(old, new, err_msg='plane %d' % (plane - 1))

                new = hits[hits['plane'] + 1 == plane]['frame']
                old = hits_old[np.where(hits_old['plane'] == plane)]['mframe'][:new.shape[0]]
                np.testing.assert_array_equal(old, new, err_msg='plane %d' % (plane - 1))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInterpretation)
    unittest.TextTestRunner(verbosity=2).run(suite)