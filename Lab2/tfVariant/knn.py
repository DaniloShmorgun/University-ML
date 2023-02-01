import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

water_data = pd.read_csv('./glass.csv')

print(water_data)

print(water_data.head(n=20))
print(water_data.describe())
print(water_data.isnull().sum())