''' Mimosa26 raw data converter for data recorded with pymosa.
'''

from __future__ import division

import os
import logging

import numpy as np
import tables as tb
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from pyBAR_mimosa26_interpreter import raw_data_interpreter
from pyBAR_mimosa26_interpreter import plotting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


class DataInterpreter(object):
    ''' Class to provide an easy to use interface to encapsulate the interpretation and event building process.
    '''

    def __init__(self, raw_data_file, analyzed_data_file=None, create_pdf=True, trigger_data_format=2, chunk_size=1000000):
        '''
        Parameters
        ----------
        raw_data_file : string
            The filename of the input raw data file.
        analyzed_data_file : string
            The file name of the output analyzed data file.
            The file extension (.h5) may not be provided.
        create_pdf : bool
            Creates interpretation plots into one PDF file.
        trigger_data_format : integer
            Number which indicates the used trigger data format.
            0: TLU word is trigger number (not supported)
            1: TLU word is timestamp (not supported)
            2: TLU word is 15 bit timestamp + 16 bit trigger number
            Only trigger data format 2 is supported, since the event building requires a trigger timestamp in order to work reliably.
        chunk_size : integer
            Chunk size of the data when reading from file. The larger the chunk size, the more RAM is consumed.
        '''
        self._raw_data_file = raw_data_file

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

        self._raw_data_interpreter = raw_data_interpreter.RawDataInterpreter()
        # self._event_builder = event_builder.EventBuilder(chunk_size)

        # Std. settings
        # if chunk_size < 10000:
        #     raise ValueError('Please chose reasonable large chunk size')
        self.chunk_size = chunk_size
        if trigger_data_format != 2:
            raise ValueError('Trigger data format different than 2 is not yet supported. For event building a trigger timestamp is required!')

        self.trigger_data_format = trigger_data_format

        self.set_standard_settings()

    def set_standard_settings(self):
        self.create_occupancy_hist = False
        self.create_error_hist = False
        self.create_hit_table = True

    @property
    def create_occupancy_hist(self):
        return self._create_occupancy_hist

    @create_occupancy_hist.setter
    def create_occupancy_hist(self, value):
        self._create_occupancy_hist = bool(value)

    @property
    def create_error_hist(self):
        return self._create_error_hist

    @create_error_hist.setter
    def create_error_hist(self, value):
        self._create_error_hist = bool(value)

    @property
    def create_hit_table(self):
        return self._create_hit_table

    @create_hit_table.setter
    def create_hit_table(self, value):
        self._create_hit_table = bool(value)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return self

    def interpret_word_table(self):
        with tb.open_file(self._raw_data_file, 'r') as in_file_h5:
            logging.info('Interpreting raw data file %s', self._raw_data_file)
            logging.info('Trigger data format: %s', self.trigger_data_format)
            with tb.open_file(self._analyzed_data_file, 'w') as out_file_h5:
                if self.create_hit_table:
                    hit_table = out_file_h5.create_table(
                        where=out_file_h5.root,
                        name='Hits',
                        description=raw_data_interpreter.hits_dtype,
                        title='hit_data',
                        filters=tb.Filters(
                            complib='blosc',
                            complevel=5,
                            fletcher32=False))

                if self.create_occupancy_hist:
                    occupancy_hist = np.zeros(shape=(6, 1152, 576), dtype=np.int32)  # for each plane

                if self.create_error_hist:
                    self.event_status_hist = np.zeros(shape=(6, 32), dtype=np.int32)  # for TLU and each plane

                logging.info("Interpreting raw data...")
                for i in tqdm(range(0, in_file_h5.root.raw_data.shape[0], self.chunk_size)):  # Loop over all words in the actual raw data file in chunks
                    raw_data_chunk = in_file_h5.root.raw_data.read(i, i + self.chunk_size)
                    hits = self._raw_data_interpreter.interpret_raw_data(raw_data=raw_data_chunk)

                    if self.create_hit_table:
                        hit_table.append(hits)

                    if self.create_occupancy_hist:
                        occupancy_hist += fill_occupancy_hist(hits)

                    if self.create_error_hist:
                        fill_event_status_hist(self.event_status_hist, hits)

                hits = self._raw_data_interpreter.interpret_raw_data(raw_data=None, build_all_events=True)

                if self.create_hit_table:
                    hit_table.append(hits)

                if self.create_occupancy_hist:
                    occupancy_hist += fill_occupancy_hist(hits)

                if self.create_error_hist:
                    fill_event_status_hist(self.event_status_hist, hits)

                # Add histograms to data file and create plots
                for plane in range(6):
                    # store occupancy map for all Mimosa26 planes
                    logging.info('Store histograms and create plots for plane %d', plane)

                    if self.create_occupancy_hist:
                        occupancy_array = out_file_h5.create_carray(
                            where=out_file_h5.root,
                            name='HistOcc_plane%d' % plane,
                            title='Occupancy Histogram of Mimosa plane %d' % plane,
                            atom=tb.Atom.from_dtype(occupancy_hist[plane].dtype),
                            shape=occupancy_hist[plane].shape,
                            filters=tb.Filters(
                                complib='blosc',
                                complevel=5,
                                fletcher32=False))
                        occupancy_array[:] = occupancy_hist[plane]
                        try:
                            if self.output_pdf:
                                plotting.plot_fancy_occupancy(hist=occupancy_hist[plane].T,
                                                              title='Occupancy for plane %d' % plane,
                                                              z_max='median',
                                                              filename=self.output_pdf)
                        except Exception:
                            logging.warning('Could not create occupancy map plot!')

                    if self.create_error_hist:
                        # plot event status histograms
                        try:
                            if self.output_pdf:
                                n_words = np.sum(self.event_status_hist[plane].T)
                                plotting.plot_event_status(hist=self.event_status_hist[plane].T,
                                                           title='Event status for plane %d ($\Sigma = % i$)' % (plane, n_words),
                                                           filename=self.output_pdf)
                        except Exception:
                            logging.warning('Could not create event status plot!')

                if self.output_pdf:
                    logging.info('Closing output PDF file: %s', str(self.output_pdf._file.fh.name))
                    self.output_pdf.close()


@njit
def fill_occupancy_hist(hits):
    hist = np.zeros(shape=(6, 1152, 576), dtype=np.int32)  # for each plane
    for hit_index in range(hits.shape[0]):
        col = hits[hit_index]['column']
        row = hits[hit_index]['row']
        plane_id = hits[hit_index]['plane']
        hist[plane_id, col, row] += 1
    return hist


@njit
def fill_event_status_hist(hist, hits):
    for hit_index in range(hits.shape[0]):
        event_status = hits[hit_index]['event_status']
        plane = hits[hit_index]['plane']
        for i in range(32):
            if event_status & (1 << i):
                hist[plane][i] += 1
