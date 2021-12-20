### import the needed libraries
import numpy as np
import orchest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Retrieve the data from the previous step.
data = orchest.get_inputs()
train, test, target = data["split_data"]

preds = [x for x in list(train) if x not in target]

print("Fitting the model...")

# Fit model. Choose the right model based on target variable type
if train[target[0]].dtype == float:
    model = RandomForestRegressor()
    modeltype = 'Regression'
else:
    model = RandomForestClassifier()
    modeltype = 'Classification'
    
## Train the model on train data
model.fit(train[preds], train[target])

# Make a prediction and send the outputs
y_test_prediction = model.predict(test[preds])
print("Success!", y_test_prediction)

### leave the space in the end as is - don't change it!
orchest.output((test[target].values, y_test_prediction), name="random_forest "+modeltype)

