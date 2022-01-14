from matplotlib import pyplot as plt
import orchest
import pandas as pd
import numpy as np

## load some input arguments for deep_autoviml 
keras_model_type = orchest.get_step_param('keras_model_type')
project_name = orchest.get_step_param("project_name")
print(keras_model_type, project_name)

## import deep_autoviml now
from deep_autoviml import deep_autoviml as deepauto

# Retrieve the data from the previous step.
data = orchest.get_inputs()
train, test, target = data["split_data"]
print('Target = %s' %target)
print(train.shape)
train.head(1)

print('Running deep_autoviml...')
model, cat_vocab_dict = deepauto.fit(train, target, keras_model_type=keras_model_type,
		project_name=project_name, keras_options={'early_stopping':True},  
		model_options={}, save_model_flag=True, use_my_model='',
		model_use_case='', verbose=1)

## load the saved model path for use later
model_or_model_path = cat_vocab_dict['saved_model_path']
print(model_or_model_path)

## send the output to make predictions in next step
orchest.output((test, model_or_model_path, project_name, keras_model_type, cat_vocab_dict, target), name="training_model_artifacts")
print("Success!")

      