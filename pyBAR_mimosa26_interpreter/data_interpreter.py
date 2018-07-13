''' Class to convert Mimosa26 raw data recorded with pymosa to hit maps.
    Additional, events can be build from the hit table using the TLU data words and time reference data.
'''

from __future__ import division

import os
import numpy as np
import tables as tb
import logging

from numba import njit
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from pyBAR_mimosa26_interpreter import raw_data_interpreter
from pyBAR_mimosa26_interpreter import event_builder
from pyBAR_mimosa26_interpreter import plotting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


@njit
def fill_occupancy_hist(hits):
    hist = np.zeros(shape=(6, 1152, 576), dtype=np.int32)  # for each plane
    for hit_index in range(hits.shape[0]):
        col = hits[hit_index]['column']
        row = hits[hit_index]['row']
        plane_id = hits[hit_index]['plane'] - 1
        hist[plane_id][col, row] += 1

    return hist


@njit
def fill_event_status_hist(hist, hits):
    for hit_index in range(hits.shape[0]):
        if hits[hit_index]['plane'] == 255:  # TLU data
            for i in range(32):
                if hits[hit_index]['event_status'] & (0b1 << i):
                    hist[0][i] += 1
        else:  # M26 data
            for i in range(32):
                plane_id = hits[hit_index]['plane']
                if hits[hit_index]['event_status'] & (0b1 << i):
                    hist[plane_id][i] += 1


class DataInterpreter(object):
    ''' Class to provide an easy to use interface to encapsulate the interpretation process.'''

    def __init__(self, raw_data_file, time_reference_file=None, analyzed_data_file=None, create_pdf=True, trigger_data_format=2, chunk_size=5000000):
        '''
        Parameters
        ----------
        raw_data_file : string or tuple, list
            A string with the raw data file name. File ending (.h5)
        analyzed_data_file : string
            The file name of the output analyzed data file. File ending (.h5)
            Does not have to be set.
        create_pdf : boolean
            Creates interpretation plots into one PDF file.
        trigger_data_format : integer
            Number which indicates the used trigger data format.
            0: TLU word is trigger number (not supported)
            1: TLU word is timestamp (not supported)
            2: TLU word is 15 bit timestamp + 16 bit trigger number
            Only trigger data format 2 is supported, since the event building requires a trigger timestamp in order to work reliably.
        chunk_size : integer
            How many raw data words are analyzed at once in RAM. Limited by available RAM. Faster
            interpretation for larger numbers. RAM needed is approximately 10 * chunk_size in bytes.
        '''

        if chunk_size < 100:
            raise RuntimeError('Please chose reasonable large chunk size')

        self._raw_data_file = raw_data_file

        self._time_reference_file = time_reference_file

        if analyzed_data_file:
            if os.path.splitext(analyzed_data_file)[1].strip().lower() != ".h5":
                self._analyzed_data_file = os.path.splitext(analyzed_data_file)[0] + ".h5"
            else:
                self._analyzed_data_file = analyzed_data_file
        else:
            self._analyzed_data_file = os.path.splitext(self._raw_data_file)[0] + '_interpreted.h5'

        if create_pdf:
            output_pdf_filename = os.path.splitext(self._raw_data_file)[0] + ".pdf"
            logging.info('Opening output PDF file: %s', output_pdf_filename)
            self.output_pdf = PdfPages(output_pdf_filename)
        else:
            self.output_pdf = None

        self._raw_data_interpreter = raw_data_interpreter.RawDataInterpreter(trigger_data_format=trigger_data_format)
        self._event_builder = event_builder.EventBuilder(chunk_size)

        # Std. settings
        self.chunk_size = chunk_size
        if trigger_data_format != 2:
            logging.warning('Trigger data format different than 2 is not supported. For event building a trigger timestamp is required!')
            raise

        self.trigger_data_format = trigger_data_format

        self.set_standard_settings()

    def set_standard_settings(self):
        self.create_occupancy_hist = True
        self.create_error_hist = True
        self.create_hit_table = False
        self._filter_table = tb.Filters(complib='blosc', complevel=5, fletcher32=False)

    @property
    def create_occupancy_hist(self):
        return self._create_occupancy_hist

    @create_occupancy_hist.setter
    def create_occupancy_hist(self, value):
        self._create_occupancy_hist = value

    @property
    def create_hit_table(self):
        return self._create_hit_table

    @create_hit_table.setter
    def create_hit_table(self, value):
        self._create_hit_table = value

    @property
    def create_error_hist(self):
        return self._create_error_hist

    @create_error_hist.setter
    def create_error_hist(self, value):
        self._create_error_hist = value

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self

    def interpret_word_table(self):
        with tb.open_file(self._raw_data_file, 'r') as in_file_h5:
            logging.info('Interpreting raw data file %s', self._raw_data_file)
            logging.info('Trigger data format: %s', self.trigger_data_format)
            with tb.open_file(self._analyzed_data_file, 'w') as out_file_h5:
                description = np.zeros((1, ), dtype=raw_data_interpreter.hit_dtype).dtype

                if self.create_hit_table:
                    hit_table = out_file_h5.create_table(out_file_h5.root,
                                                         name='Hits',
                                                         description=description,
                                                         title='hit_data',
                                                         filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                         chunkshape=(self.chunk_size / 100,))

                if self.create_occupancy_hist:
                    occupancy_hist = np.zeros(shape=(6, 1152, 576), dtype=np.int32)  # for each plane

                if self.create_error_hist:
                    self.event_status_hist = np.zeros(shape=(7, 32), dtype=np.int32)  # for TLU and each plane

                logging.info("Interpreting raw data...")
                for i in tqdm(range(0, in_file_h5.root.raw_data.shape[0], self.chunk_size)):  # Loop over all words in the actual raw data file in chunks
                    raw_data_chunk = in_file_h5.root.raw_data.read(i, i + self.chunk_size)
                    hits = self._raw_data_interpreter.interpret_raw_data(raw_data_chunk)

                    if self.create_hit_table:
                        hit_table.append(hits)

                    if self.create_occupancy_hist:
                        occupancy_hist += fill_occupancy_hist(hits[hits['plane'] != 255])

                    if self.create_error_hist:
                        fill_event_status_hist(self.event_status_hist, hits)
                        print self.event_status_hist

                # Add histograms to data file and create plots
                for plane in range(7):
                    if plane == 0:  # TLU, only store event status histogram
                        logging.info('Store histograms and create plots for TLU')

                        if self.create_error_hist:
                            # plot event status histograms
                            try:
                                if self.output_pdf:
                                    n_words = np.sum(self.event_status_hist[plane].T)
                                    plotting.plot_event_status(hist=self.event_status_hist[plane].T,
                                                               title='Event status for plane %d ($\Sigma = % i$)' % (plane, n_words),
                                                               filename=self.output_pdf)
                            except:
                                logging.warning('Could not create event status plot!')
                    else:
                        # store occupancy map for all Mimosa26 planes
                        logging.info('Store histograms and create plots for plane %d', plane)

                        if self.create_occupancy_hist:
                            occupancy_array = out_file_h5.create_carray(out_file_h5.root, name='HistOcc_plane%d' % plane,
                                                                        title='Occupancy Histogram of Mimosa plane %d' % plane,
                                                                        atom=tb.Atom.from_dtype(occupancy_hist[plane - 1].dtype),
                                                                        shape=occupancy_hist[plane - 1].shape, filters=self._filter_table)
                            occupancy_array[:] = occupancy_hist[plane - 1]
                            try:
                                if self.output_pdf:
                                    plotting.plot_fancy_occupancy(occupancy_hist[plane - 1].T, z_max='median',
                                                                  title='Occupancy for plane %d' % plane, filename=self.output_pdf)
                            except:
                                logging.warning('Could not create occupancy map plot!')

                        if self.create_error_hist:
                            # plot event status histograms
                            try:
                                if self.output_pdf:
                                    n_words = np.sum(self.event_status_hist[plane].T)
                                    plotting.plot_event_status(hist=self.event_status_hist[plane].T,
                                                               title='Event status for plane %d ($\Sigma = % i$)' % (plane, n_words),
                                                               filename=self.output_pdf)
                            except:
                                logging.warning('Could not create event status plot!')

                if self.output_pdf:
                    logging.info('Closing output PDF file: %s', str(self.output_pdf._file.fh.name))
                    self.output_pdf.close()

    def interpret_hit_table(self):
        if self._time_reference_file is None:
            logging.error('No data file for time reference plane specified. Cannot build events from hit table!')
            raise

        # First step: build events from interpreted hit table for each plane
        for plane in range(1, 7):
            logging.info("Building events for plane %i...", plane)
            self.build_events_from_hit_table(input_file=self._analyzed_data_file,
                                             output_file=self._analyzed_data_file[:-3] + '_event_build_plane_%i.h5' % plane,
                                             plane=plane,
                                             chunk_size=self.chunk_size)

        # Second step: align events with time reference plane
        for plane in range(1, 7):
            logging.info("Aligning data of plane %i with time reference...", plane)
            self.align_with_time_reference(input_file=self._analyzed_data_file[:-3] + '_event_build_plane_%i.h5' % plane,
                                           input_file_time_reference=self._time_reference_file,
                                           output_file=self._analyzed_data_file[:-3] + '_event_build_aligned_plane_%i.h5' % plane,
                                           chunk_size=int(self.chunk_size / 10))  # use smaller chunksize, otherwise correlation buffer needs to be too large

    def build_events_from_hit_table(self, input_file, output_file, plane, chunk_size=5000000):
        '''
        Build events from M26 hit table using TLU data words. One TLU data word is assigned to one M26 data frame.
        '''

        # reset variables before event building after each plane
        self._event_builder.reset()
        last_chunk = False  # indicates last chunk
        description = np.zeros((1, ), dtype=self._event_builder.event_table_dtype).dtype

        with tb.open_file(output_file, 'w') as out_file_h5:
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=description,
                                                     title='Hit Table for Testbeam Analysis',
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            with tb.open_file(input_file, 'r') as in_file_m26_h5:
                m26_hit_table = in_file_m26_h5.root.Hits[:]
                n_m26 = m26_hit_table.shape[0]
                for i in tqdm(range(0, n_m26, chunk_size)):
                    hits = m26_hit_table[i:i + chunk_size]
                    if i + chunk_size > n_m26:
                        last_chunk = True  # set last chunk indicator

                    hit_data_out = self._event_builder.build_events(hits,
                                                                    plane,
                                                                    last_chunk)

                    # append to table
                    hit_table_out.append(hit_data_out)
                    hit_table_out.flush()

    def align_with_time_reference(self, input_file, input_file_time_reference, output_file, chunk_size=500000):
        '''
        Align M26 data with time reference data.
        '''

        with tb.open_file(input_file_time_reference, 'r') as in_file_ref_h5:
            hit_table_time_reference = in_file_ref_h5.root.Hits[:]

        with tb.open_file(output_file, 'w') as out_file_h5:
            description = np.zeros((1, ), dtype=self._event_builder.event_table_dtype).dtype
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits',
                                                     description=description, title='hit_data',
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            with tb.open_file(input_file, 'r') as in_file_h5:
                m26_hit_table = in_file_h5.root.Hits[:]
                n_m26 = m26_hit_table.shape[0]
                n_time_reference = hit_table_time_reference.shape[0]
                for i in tqdm(range(0, n_m26, chunk_size)):
                    m26_data = m26_hit_table[i:i + chunk_size]
                    ratio = n_time_reference / n_m26  # needed for properly chunking of time reference data with respect to chunked m26 data
                    reference_data = hit_table_time_reference[int(i * ratio):int((i + chunk_size) * ratio)][["event_number", "trigger_number"]]

                    # align data with time reference
                    hit_buffer = self._event_builder.align_with_time_ref(m26_data, reference_data)

                    # append to table
                    hit_table_out.append(hit_buffer)
                    hit_table_out.flush()
