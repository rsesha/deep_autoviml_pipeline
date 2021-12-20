import orchest
from deep_autoviml import print_classification_model_stats, print_regression_model_stats

## just collect the results from various predictions of models
data1 = orchest.get_inputs()

### print the results one by one ####
for name, value in data1.items():
    if name != "unnamed":
        modelname = name.split(" ")[:-1]
        modeltype =  name.split(" ")[-1]
        print(f"\n{modelname} ")
        y_test, y_preds = value
        if modeltype == 'Regression':
            print_regression_model_stats(y_test, y_preds)
        else:
            print_classification_model_stats(y_test, y_preds)