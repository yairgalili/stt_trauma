from allosaurus.app import read_recognizer
import csv
import os
import glob
import ipapy
from ipapy.arpabetmapper import ARPABETMapper
from Levenshtein import distance

# load your model
model = read_recognizer()

# run inference -> æ l u s ɔ ɹ s
sample_rate_list = range(10000,30000,1000)
# a = model.recognize(r"C:\Users\yair\stt_trauma\data\tach\tach-1.wav",lang_id='eng',sample_rate_list = [22000,16000])

def remove_non_ascii(string):
    return string.encode('ascii', errors='ignore').decode()

def command_u2a(string):
    """
    Print the ARPABEY ASCII string corresponding to the given Unicode IPA string.

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    l = ARPABETMapper().map_unicode_string(
        unicode_string=string,
        ignore=True,
        single_char_parsing=True,
        return_as_list=False
    )
    return l

true_opcodes = ["SHUKI","TAHI"]
#
data_folder = r'C:\Users\yair\stt_trauma\data'
    
header = ['filename', *[f'sample_rate_{i}' for i in sample_rate_list],'label', *[f'distance_{x}' for x in true_opcodes]]

# The weights for the three operations in the form (insertion, deletion, substitution). 
unwanted_char = 'W'

with open(os.path.join(data_folder,'data_allosaurus_with_u2a_filter_distances_012_try_2.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    types_list = 'tach bla'.split()
    for g in types_list:
        for filename in glob.glob(os.path.join(data_folder,g,'*.wav')):
            a = model.recognize(filename,lang_id='eng',sample_rate_list = sample_rate_list)
            print(a)
            text_after_u2a = [command_u2a(e) for e in a]
            to_append = [filename,
                         *text_after_u2a, g]
            for x in true_opcodes:
                # The weights for the three operations in the form (insertion, deletion, substitution). 
                to_append.append(min([distance(x, a, weights=(0,1,2))
                                      for a in text_after_u2a]))                
            writer = csv.writer(file)
            writer.writerow(to_append)






