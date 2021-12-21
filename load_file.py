import orchest
import pandas as pd
datapath = "data/"
filename = orchest.get_step_param('filename')
sep =  orchest.get_step_param('sep')
target = orchest.get_step_param('target')

# Explicitly cache the data in the "/data" directory since the
# kernel is running in a Docker container, which are stateless.
# The "/data" directory is a special directory managed by Orchest
# to allow data to be persisted and shared across pipelines and
# even projects.
print("Loading file from data folder...")
df_data = pd.read_csv(datapath+filename, sep=sep, header=0)
if isinstance(target, str):
    targetvar = [target]
else:
    targetvar = target

# Convert the data into a DataFrame.

# Output the housing data so the next steps can retrieve it.
print("Outputting converted housing data...")
orchest.output((df_data, targetvar), name="data")
print("Success!")
