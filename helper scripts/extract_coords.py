import csv
import os
import pandas as pd


dir_cwg = 'data/soundcomparisons/cwg/'
out_path = 'helper scripts/coordinates.csv'

fo = open(out_path, 'w', encoding='utf8', newline='')
writer = csv.writer(fo, delimiter=',')
writer.writerow(['longitude', 'latitude', 'doculect'])
for root, dirs, files in os.walk(dir_cwg):
    for f in files:
        if f.endswith('.csv') and 'ProtoGermanic' not in f:
            df = pd.read_csv(dir_cwg + f, encoding='utf8')
            longitude = df['Longitude'][0]
            latitude = df['Latitude'][0]
            doculect = f[:-4]
            writer.writerow([longitude, latitude, doculect])
fo.close()
