Files bcnn_DD_woft.py and bcnn_DD_woft_with_random_crops.py are used 
for the first step of training procedure where only last layer of the BCNN_DD model is trained.
--> Learning rate = 0.9
--> Optimizer = Momentum optimizer with 0.9 momentum

Files bcnn_finetuning.py and bcnn_finetuning_with_random_crops.py are used 
for the second step of training procedure where finetuning is performed on the
entire BCNN_DD model.
--> Learning rate = 0.001
--> Optimizer = Momentum optimizer with 0.9 momentum

