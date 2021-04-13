import argparse
import csv
import matplotlib.pyplot as plt
import pandas as pd

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train EDRV GAN: Super Resolution Models')
parser.add_argument('--csv_path', help='path to csv file')

opt = parser.parse_args()

file_path = opt.csv_path
data = pd.read_csv(file_path)

for column in data:
  if column == 'Epoch':
    continue
  
  df = pd.DataFrame(data,columns=[column,'Epoch'])
  df.plot(x ='Epoch', y=column)
  plt.show()
