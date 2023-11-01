import os
import numpy as np
import pandas as pd

from data_preprocess import process_raw_data, editnet_data_to_editnetID

input_name = 'test'

data_comp = open('./dataset/{}_comp.txt'.format(input_name), "r", encoding="utf-8").read().splitlines()
data_simp = open('./dataset/{}_simp.txt'.format(input_name), "r", encoding="utf-8").read().splitlines()

df = process_raw_data(data_comp, data_simp)

print(df.head())
print()

editnet_data_to_editnetID(df, "./outputs/{}.df.filtered.pos".format(input_name))

