'''Example how to use the M26 data interpreter. A hit table is created from raw data, additionally events are build using the TLU data words.
The example raw data file was taken with pyBAR, thats why also FE-I4 data is included in the raw data and the event status for the TLU
displays unkown_words (FE-I4 words, only M26 and TLU data is expected in the raw data).
'''

import logging

from multiprocessing import Pool

from pyBAR_mimosa26_interpreter import data_interpreter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def analyze_raw_data(input_file, time_reference_file, trigger_data_format):  # Mimosa26 raw data analysis
    with data_interpreter.DataInterpreter(raw_data_file=input_file, time_reference_file=time_reference_file, trigger_data_format=trigger_data_format) as raw_data_analysis:
        raw_data_analysis.create_hit_table = True
        raw_data_analysis.interpret_word_table()
        raw_data_analysis.interprete_hit_table()


def process_dut(raw_data_file, time_reference_file, trigger_data_format):
    analyze_raw_data(raw_data_file, time_reference_file, trigger_data_format=trigger_data_format)

# def format_hit_table(input_file, output_file):
#     ''' Selects and renames important columns for test beam analysis and stores them into a new file.
# 
#     Parameters
#     ----------
#     input_file : pytables file
#     output_file : pytables file
#     '''
# 
#     logging.info('Format hit table in %s', input_file)
#     with tb.open_file(input_file, 'r') as in_file_h5:
#         hits = in_file_h5.root.Hits[:]
#         hits_formatted = np.zeros((hits.shape[0], ), dtype=[('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
#         with tb.open_file(output_file, 'w') as out_file_h5:
#             hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=hits_formatted.dtype, title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
#             hits_formatted['event_number'] = hits['event_number']
#             hits_formatted['frame'] = hits['relative_BCID']
#             hits_formatted['column'] = hits['row']
#             hits_formatted['row'] = hits['column']
#             hits_formatted['charge'] = hits['tot']
#             if not np.all(np.diff(hits_formatted['event_number']) >= 0):
#                 raise RuntimeError('The event number does not always increase. This data cannot be used like this!')
#             hit_table_out.append(hits_formatted)


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
