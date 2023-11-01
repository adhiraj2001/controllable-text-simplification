import csv
import pandas as pd 

input_file = 'test.tsv'

colnames = ['source', 'target']
df = pd.read_csv(input_file, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', header=None, names=colnames)

with open(f"{input_file.rsplit('.', 1)[0]}_comp.txt", 'w') as f:
    f.write('\n'.join(df['source'].astype(str)))

with open(f"{input_file.rsplit('.', 1)[0]}_simp.txt", 'w') as f:
    f.write('\n'.join(df['target'].astype(str)))
