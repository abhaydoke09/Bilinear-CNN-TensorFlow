Bilinear CNN

This is an implementation of Bilinear
Convolutional Neural Network (Bilinear CNN) using TensorFlow.

Main objective of this project is to implement Bilinear
Convolutional Neural Network (Bilinear CNN) for Fine-grained Visual Recognition using
TensorFlow. I implemented the Bilinear
Convolutional Neural Network (Bilinear CNN) model as
described in the http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
 and trained it on the FGVC-Aircraft
dataset with 100 categories. Bilinear
Convolutional Neural Network model combines
two Convolutional Neural Network architectures pre-trained on the ImageNet dataset
using outer product at each location in the image. Training
Bilinear Convolutional Neural Network  model is a two step training procedure in which
the last fully connected layer is trained first followed by
the fine-tuning entire model using back propagation. In
this project, I present experimental results of two methods
on top of the Bilinear CNN (DD) model as described in http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
which uses two VGG16 models pretrained on ImageNet
dataset and then Bilinear CNN (DD) model is trained on the FGVCAircraft
dataset. One, I experimented with a slightly different
approach in the two-step training procedure, where
the training of the last layer is interrupted after 10-15
epochs and fine-tuning the entire model was started after
that. Two, I used random cropping of images during the
training of Bilinear CNN (DD) model to see if there is any significant
improvement in the accuracy of the Bilinear CNN (DD) model
on the FGVC-Aircraft dataset. I obtain 86.4% accuracy
with the first method and 85.41% accuracy with the second
method. Training of the network is done on NVIDIA Tesla
M40 GPU. Training of last layer of Bilinear CNN (DD) TensorFlow model runs
at 20 frames/second and fine-tuning the entire Bilinear CNN (DD) TensorFlow model runs
at 10 frames/second.


To download the VGG16 model weigths and to get the TensorFlow model for VGG16, go to -> https://www.cs.toronto.edu/~frossard/post/vgg16/

I provide the BCNN implmentation in TensorFlow. 

bcnn_DD_woft.py and bcnn_DD_woft_with_random_crops.py are TensorFlow files used 
for the first step of the training procedure where only last layer of the Bilinear CNN (DD) model is trained.</br>
--> Learning rate = 0.9</br>
--> Optimizer = Momentum optimizer with 0.9 momentum</br>

bcnn_finetuning.py and bcnn_finetuning_with_random_crops.py are TensorFlow files used 
for the second step of the training procedure where finetuning is performed on the
entire Bilinear CNN (DD) model.</br>
--> Learning rate = 0.001</br>
--> Optimizer = Momentum optimizer with 0.9 momentum</br>

