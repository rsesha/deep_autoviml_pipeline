import orchest
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Retrieve the data from the previous step.
data = orchest.get_inputs()  # data = [(df_data, df_target)]
data, target = data["data"]

# Print messages are useful when you are keeping an eye on the logs of
# a pipeline step.
print("Splitting the data into train and test...")
#### we are going to infer what type of problem it is by looking at the target variable ##
if data[target[0]].dtype == float:
    train, test = train_test_split(data, test_size=0.2, random_state=9)
else:
    train, test = train_test_split(data, test_size=0.2, random_state=9, stratify=data[target])
print(train.shape, test.shape)

orchest.output((train, test, target), name="split_data")
print("Success!")