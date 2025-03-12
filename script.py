import numpy as np
import pandas as pd

def load_data(filepath):
  data = pd.read_csv(filepath)
  x = data.iloc[:, :-1]
  y = data.iloc[:, -1]

  # Preprocess data
  quantitative_data  = x.columns[x.dtypes == "int64"]


def main():
  filepath = "Thyroid_Diff.csv"


  
if __name__ == "__main__":
  main()