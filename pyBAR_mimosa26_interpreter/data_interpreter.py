''' Mimosa26 raw data converter for data recorded with pymosa.
'''

from __future__ import division

import os
import logging

import numpy as np
import tables as tb
from numba import njit
from tqdm import tqdm
try:
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    pass

from pyBAR_mimosa26_interpreter import raw_data_interpreter
try:
    from pyBAR_mimosa26_interpreter import plotting
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


class DataInterpreter(object):
    ''' Class to provide an easy to use interface to encapsulate the interpretation and event building process.
    '''

    def __init__(self, raw_data_file, analyzed_data_file=None, trigger_data_format=2, add_missing_events=False, timing_offset=None, active_m26_planes=[1, 2, 3, 4, 5, 6], create_pdf=False, chunk_size=1000000):
        '''
        Parameters
        ----------
        raw_data_file : string
            The filename of the input raw data file.
        analyzed_data_file : string
            The file name of the output analyzed data file.
            The file extension (.h5) may not be provided.
        trigger_data_format : integer
            Number which indicates the used trigger data format.
            0: TLU word is trigger number (not supported)
            1: TLU word is timestamp (not supported)
            2: TLU word is 15 bit timestamp + 16 bit trigger number
            Only trigger data format 2 is supported, since the event building requires a trigger timestamp in order to work reliably.
        add_missing_events : boolean
            If True, add (silently) missing events (due to missing trigger words). Default is False.
        timing_offset : int
            Offset between Mimosa26 40 MHz clock and 40 MHz from R/O system. If None, use default value which was obtained
            by maximizing correlation between Mimosa26 telescope and time reference.
        active_m26_planes : list
            List of Mimosa26 plane header IDs that will be interpreted. Default: [1, 2, 3, 4, 5, 6].
        create_pdf : bool
            If True, create PDF containing several ouput plots.
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

        if self._raw_data_file == self._analyzed_data_file:
            raise ValueError('Filename of the input and output file must be different.')

        self.output_pdf = None
        if create_pdf:
            output_pdf_filename = os.path.splitext(self._raw_data_file)[0] + ".pdf"
            try:
                self.output_pdf = PdfPages(output_pdf_filename)
            except NameError:
                create_pdf = False
            else:
                logging.info('Opening output PDF file: %s' % output_pdf_filename)

        self._raw_data_interpreter = raw_data_interpreter.RawDataInterpreter()
        if add_missing_events is not None:
            self._raw_data_interpreter.add_missing_events = add_missing_events
        if timing_offset is not None:
            self._raw_data_interpreter.timing_offset = timing_offset

        # Std. settings
        self.chunk_size = chunk_size
        if trigger_data_format != 2:
            raise ValueError('Trigger data format different than 2 is not yet supported. For event building a trigger timestamp is required!')

        self.set_standard_settings()

        self.active_m26_planes = active_m26_planes
        logging.info('Interpreting the following Mimosa26 planes with header IDs: %s' % ', '.join(active_m26_planes))

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
            logging.info('Interpreting raw data file %s...' % self._raw_data_file)
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
                    hits = self._raw_data_interpreter.interpret_raw_data(raw_data=raw_data_chunk, active_m26_planes=self.active_m26_planes)

                    if self.create_hit_table:
                        hit_table.append(hits)
                        hit_table.flush()

                    if self.create_occupancy_hist:
                        occupancy_hist += fill_occupancy_hist(hits)

                    if self.create_error_hist:
                        fill_event_status_hist(self.event_status_hist, hits)

                # get last incomplete events
                hits = self._raw_data_interpreter.interpret_raw_data(raw_data=None, build_all_events=True, active_m26_planes=self.active_m26_planes)

                if self.create_hit_table:
                    hit_table.append(hits)

                if self.create_occupancy_hist:
                    occupancy_hist += fill_occupancy_hist(hits)

                if self.create_error_hist:
                    fill_event_status_hist(self.event_status_hist, hits)

                # Add histograms to data file and create plots
                for plane in self.active_m26_planes:
                    # store occupancy map for all Mimosa26 planes
                    logging.info('Store histograms %sfor Mimosa26 plane with header ID %d.' % ('and create plots ' if self.output_pdf else '', plane))

                    if self.create_occupancy_hist:
                        occupancy_array = out_file_h5.create_carray(
                            where=out_file_h5.root,
                            name='HistOcc_plane%d' % plane,
                            title='Occupancy histogram for Mimosa26 plane with header ID %d' % plane,
                            atom=tb.Atom.from_dtype(occupancy_hist[plane].dtype),
                            shape=occupancy_hist[plane].shape,
                            filters=tb.Filters(
                                complib='blosc',
                                complevel=5,
                                fletcher32=False))
                        occupancy_array[:] = occupancy_hist[plane]
                        if self.output_pdf:
                            try:
                                plotting.plot_fancy_occupancy(
                                    hist=occupancy_hist[plane].T,
                                    title='Occupancy histogram for Mimosa26 plane with header ID %d' % plane,
                                    z_max='median',
                                    filename=self.output_pdf)
                            except Exception:
                                logging.warning('Could not create occupancy plot!')

                    if self.create_error_hist:
                        # plot event status histograms
                        if self.output_pdf:
                            try:
                                n_words = np.sum(self.event_status_hist[plane].T)
                                plotting.plot_event_status(
                                    hist=self.event_status_hist[plane].T,
                                    title='Event status for Mimosa26 plane with header ID %d ($\Sigma = % i$)' % (plane, n_words),
                                    filename=self.output_pdf)
                            except Exception:
                                logging.warning('Could not create event status plot!')

                if self.output_pdf:
                    logging.info('Closing output PDF file: %s' % self.output_pdf._file.fh.name)
                    try:
                        self.output_pdf.close()
                    except Exception:
                        pass


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
