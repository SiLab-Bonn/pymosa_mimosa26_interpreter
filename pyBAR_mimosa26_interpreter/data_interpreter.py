''' Class to convert Mimosa 26 raw data recorded with pyBAR to hit maps.
'''
import os
import numpy as np
import tables as tb
import logging
import progressbar
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages

from pyBAR_mimosa26_interpreter import raw_data_interpreter
from pyBAR_mimosa26_interpreter import plotting

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


@njit
def fill_occupanc_hist(hist, hits):
    for hit_index in range(hits.shape[0]):
        if hits[hit_index]['plane']==0:
            pass  # tlu data in plane=0 
        else:
            hist[hits[hit_index]['plane']-1][hits[hit_index]['column'], hits[hit_index]['row']] += 1


class DataInterpreter(object):
    ''' Class to provide an easy to use interface to encapsulate the interpretation process.'''

    def __init__(self, raw_data_file, analyzed_data_file=None, create_pdf=True, chunk_size=5000000): 
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
        chunk_size : integer
            How many raw data words are analyzed at once in RAM. Limited by available RAM. Faster
            interpretation for larger numbers. RAM needed is approximately 10 * chunk_size in bytes.
        '''

        if chunk_size < 100:
            raise RuntimeError('Please chose reeasonable large chunk size')

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

        # Std. settings
        self.chunk_size = chunk_size

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
        if self.output_pdf:
            logging.info('Closing output PDF file: %s', str(self.output_pdf._file.fh.name))
            self.output_pdf.close()

    def interpret_word_table(self):
        with tb.open_file(self._raw_data_file, 'r') as in_file_h5:
            logging.info('Interpreting raw data file %s', self._raw_data_file)
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
                    self.occupancy_arrays = np.zeros(shape=(6, 1152, 576), dtype=np.int32)

                logging.info("Interpreting...")
                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.raw_data.shape[0], term_width=80)
                progress_bar.start()
                for word_index in range(0, in_file_h5.root.raw_data.shape[0], self.chunk_size):  # Loop over all words in the actual raw data file in chunks
                    raw_data_chunk = in_file_h5.root.raw_data.read(word_index, word_index + self.chunk_size)
                    hits = self._raw_data_interpreter.interpret_raw_data(raw_data_chunk)

                    if self.create_hit_table:
                        hit_table.append(hits)

                    if self.create_occupancy_hist:
                        fill_occupanc_hist(self.occupancy_arrays, hits)
                    progress_bar.update(word_index)
                progress_bar.finish()

                # Add histograms to data file and create plots
                for plane in range(6):
                    logging.info('Store histograms and create plots for plane %d', plane)
                    occupancy_array = out_file_h5.createCArray(out_file_h5.root, name='HistOcc_plane%d' % plane, title='Occupancy Histogram of Mimosa plane %d' % plane, atom=tb.Atom.from_dtype(self.occupancy_arrays[plane].dtype), shape=self.occupancy_arrays[plane].shape, filters=self._filter_table)
                    occupancy_array[:] = self.occupancy_arrays[plane]
                    if self.output_pdf:
                        plotting.plot_fancy_occupancy(self.occupancy_arrays[plane].T, title='Occupancy for plane %d' % plane, filename=self.output_pdf)
