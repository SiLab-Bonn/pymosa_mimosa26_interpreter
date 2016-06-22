''' Example how to interpret raw data and how to histogram the hits.
'''

import tables as tb
import sys

from pyBAR_mimosa26_interpreter import data_interpreter

# Example 1: How to use the build_hits function directly
#with tb.open_file('example_data_2.h5', 'r') as in_file_h5:
#    raw_data = in_file_h5.root.raw_data.read(8234, 8234 + 636 + 1000)
#    hits = data_interpreter.build_hits(raw_data)
fname=sys.argv[1]
print fname
# Example 2: How to use the interpretation class to convert a raw data tabe
with data_interpreter.DataInterpreter(raw_data_file=fname) as raw_data_analysis:
    raw_data_analysis.interpret_word_table()

