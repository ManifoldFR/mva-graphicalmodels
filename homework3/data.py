"""
Process the data
"""
import pandas as pd

data = pd.read_table("german.data-numeric", sep="  ", header=None)

print(data)
