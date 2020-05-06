import json
import pandas
import math
import sys
import os
import numpy as np
import re
import shutil

input_directory = "../data/indices"

# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

for file_name in csv_input_files:
    textfile = open(file_name)
    lines = textfile.readlines()

    tmp_lines = []

    for line in reversed(lines):
        tmp_lines.append(line)

    textfile.close()



    textfile = open(file_name, "w+")
    textfile.write(tmp_lines[-1])
    textfile.write(tmp_lines[0]+"\n")
    for line in range(1,len(tmp_lines)-2):
        textfile.write(tmp_lines[line])
    textfile.write(tmp_lines[-2][:-1])

    textfile.close()