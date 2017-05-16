This is an implementation of Bilinear-CNN using TensorFlow.


Files bcnn_DD_woft.py and bcnn_DD_woft_with_random_crops.py are used 
for the first step of the training procedure where only last layer of the BCNN_DD model is trained.</br>
--> Learning rate = 0.9</br>
--> Optimizer = Momentum optimizer with 0.9 momentum</br>

Files bcnn_finetuning.py and bcnn_finetuning_with_random_crops.py are used 
for the second step of the training procedure where finetuning is performed on the
entire BCNN_DD model.</br>
--> Learning rate = 0.001</br>
--> Optimizer = Momentum optimizer with 0.9 momentum</br>

