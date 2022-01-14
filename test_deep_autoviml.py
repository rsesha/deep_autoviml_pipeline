from matplotlib import pyplot as plt
import orchest

from deep_autoviml import deep_autoviml as deepauto

# Retrieve the data from the previous step.
data = orchest.get_inputs()
test, model, project_name, keras_model_type, cat_vocab_dict, target = data["training_model_artifacts"]
print('Loaded test data successfully')

modeltype = cat_vocab_dict['modeltype']

### Make predictions using deepauto library and test data #####
predictions = deepauto.predict(model, project_name, test_dataset=test,
                                 keras_model_type=keras_model_type, 
                                 cat_vocab_dict=cat_vocab_dict)

from deep_autoviml.utilities.utilities import print_classification_model_stats, print_regression_model_stats

### You need to display results from various predictions ##
import numpy as np
y_preds = np.array([])
y_pred = np.array([])

#### convert target to a list to print the results here ###
if isinstance(target, str):
    targetvar = [target]
else:
    targetvar = target

for num, each_target in enumerate(targetvar):
    if modeltype == 'Regression':
        y_pred = predictions[0]
        print_regression_model_stats(test[each_target].values, y_pred)
    else:
        y_pred = predictions[1]
        print_classification_model_stats(test[each_target].values, y_pred)
    if num == 0:
        y_preds = y_pred[:]
    else:
        y_preds = np.c_[predictions, y_pred]

print("Success!")

### Send predictions to be read by the next step ###
orchest.output((test[target].values, y_preds), name="deep_autoviml " +modeltype)


