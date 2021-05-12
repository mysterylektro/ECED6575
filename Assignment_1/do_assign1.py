"""
ECED 6575 (Underwater Acoustics Engg) - Assign 1:  Time Series Analysis
2020/2021 Summer Term
May 10, 2021

calls function(s):  assign1(data_file, data_file_arg)

students will write assign1(data_file, data_file_arg)
"""

from assign1 import assign1

# data_files = ['TRAC1_noise_time']
data_files = ['TRAC1_noise_time', 'TRAC3_sin100_time', 'TRAC11_burstnoise_time']
for data_file in data_files:
    assign1(data_file + '.csv')
