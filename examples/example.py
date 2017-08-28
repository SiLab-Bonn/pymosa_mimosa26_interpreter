''' Example how to interpret raw data and how to histogram the hits.
'''
from pyBAR_mimosa26_interpreter import data_interpreter


raw_data_file = r''

# Example: How to use the interpretation class to convert a raw data tabe
with data_interpreter.DataInterpreter(raw_data_file) as raw_data_analysis:
    raw_data_analysis.create_hit_table = True
    raw_data_analysis.interpret_word_table()
