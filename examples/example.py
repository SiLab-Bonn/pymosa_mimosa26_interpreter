'''Example how to use the M26 data interpreter. A hit table is created from raw data, additionally events are build using the TLU data words.
At the end, the hit table is formatted into the correct data format needed for testbeam analysis.
The example raw data file was taken with pyBAR, thats why also FE-I4 data is included in the raw data and the event status for the TLU
displays unkown_words (FE-I4 words, only M26 and TLU data is expected in the raw data).
'''

import logging
import numpy as np
import tables as tb

from tqdm import tqdm
from multiprocessing import Pool

from pyBAR_mimosa26_interpreter import data_interpreter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def analyze_raw_data(input_file, time_reference_file, trigger_data_format):  # Mimosa26 raw data analysis
    with data_interpreter.DataInterpreter(raw_data_file=input_file, time_reference_file=time_reference_file, trigger_data_format=trigger_data_format) as raw_data_analysis:
        raw_data_analysis.create_hit_table = True
        raw_data_analysis.interpret_word_table()  # interpret raw data
        raw_data_analysis.interpret_hit_table()  # build events


def process_dut(raw_data_file, time_reference_file, trigger_data_format, transpose=False, frame=False):
    # analyze raw data
    analyze_raw_data(raw_data_file, time_reference_file, trigger_data_format=trigger_data_format)
    # format hit table to correct data format for testbeam analysis
    for plane in range(1, 7):
        format_hit_table(input_file=raw_data_file[:-3] + '_interpreted_event_build_aligned_plane_%i.h5' % plane,
                         output_file=raw_data_file[:-3] + '_aligned_plane_%i.h5' % plane,
                         transpose=transpose,
                         frame=frame,
                         chunk_size=1000000)


def format_hit_table(input_file, output_file, transpose=False, frame=False, chunk_size=1000000):
    logging.info("Format hit table...")
    with tb.open_file(input_file, 'r') as in_file_h5:
        m26_hit_table = in_file_h5.root.Hits[:]
        n_m26 = m26_hit_table.shape[0]
        with tb.open_file(output_file, 'w') as out_file_h5:
            hits_formatted = np.zeros((n_m26, ), dtype=[('event_number', np.int64), ('frame', np.uint8),
                                                        ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits',
                                                     description=hits_formatted.dtype, title='Selected hits for test beam analysis',
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            for i in tqdm(range(0, n_m26, chunk_size)):
                m26_data = m26_hit_table[i:i + chunk_size]

                hits_formatted['event_number'] = m26_data['event_number']

                if frame:
                    hits_formatted['frame'] = m26_data['frame']
                else:
                    hits_formatted['frame'] = 0

                if transpose:
                    hits_formatted['column'] = m26_data['row'] + 1
                    hits_formatted['row'] = m26_data['column'] + 1
                else:
                    hits_formatted['column'] = m26_data['column'] + 1
                    hits_formatted['row'] = m26_data['row'] + 1

                hits_formatted['charge'] = 0

                hit_table_out.append(hits_formatted)
                hit_table_out.flush()


if __name__ == "__main__":
    # specify M26 raw data files
    raw_data_files = [r'./anemone_raw_data.h5'
                      ]
    # specify aligned time reference data files
    time_reference_files = [r'./time_reference_interpreted_data.h5'
                            ]

    trigger_data_format = 2  # use combined mode for raw data interpretation, this is the only supported format

    # pool = Pool()
    # results = pool.map(process_dut, raw_data_files, time_reference_file, trigger_data_format)
    for raw_data_file, time_reference_file in zip(raw_data_files, time_reference_files):
        process_dut(raw_data_file, time_reference_file, trigger_data_format)
    # pool.close()
    # pool.join()
