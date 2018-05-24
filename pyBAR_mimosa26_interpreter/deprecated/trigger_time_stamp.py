''' Example how to interpret raw data with trigger time stamps with Mimosa + FE-I4 telscope.

    Tested on: testbeam_analsis improvements branch that uses pybar_fei4_converter development
'''

import logging
import pyBAR_mimosa26_interpreter.simple_converter_combined as m26_cv

from testbeam_analysis.converter import pybar_fei4_converter as pyfei4_cv


def interpret_telescope_data(raw_data_file):
    ''' Runs all step to analyse telescope raw data '''

    # Step 1: Interpret FE-I4 raw data
    # Output: file with _aligned.h5 suffix
    pyfei4_cv.process_dut(raw_data_file,
                          trigger_data_format=2)  # Data format has trigger time stamp

    # Step 2: Interpret One MIMOSA26 plane
    # Output: file with _aligned.h5 suffix with plane number
    m26_cv.m26_converter(fin=raw_data_file,  # Input file
                         fout=raw_data_file[:-3] + \
                         '_frame_aligned_1.h5',  # Output file
                         plane=1)  # Plane number

    # Step 3: Combine FE with Mimosa data
    # Output: file with
    m26_cv.align_event_number(fin=raw_data_file[:-3] + '_frame_aligned_1.h5',  # Mimosa
                              fe_fin=raw_data_file[:-3] + '_event_aligned.h5',
                              fout=raw_data_file[:-3] + '_run_aligned.h5',
                              tr=True,  # Switch column / row (transpose)
                              frame=False)  # Add frame info (not working?)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    raw_data_file = r'telescope_raw_data.h5'

    interpret_telescope_data(raw_data_file)
