# Representation Learning for Image-to-Image-Translation

Deep learning has excelled in numerous computer vision applications, but they still have a
limited capacity to combine data from different domains. Because of this, performance
might be negatively impacted by even little changes in the training data. Domain shift,
the difference between observed and expected data, is crucial to studying robotics as a
slight change in algorithms can put a robot at risk while operating it physically. One
potential strategy for modifying current data to match better what an algorithm might
anticipate during operation is to use generative methods, which try to produce data
depending on the input. In this work, I researched and implemented various Generative Adversarial
Networks and Vector Quantization methods capable of generating real-world scenarios
from computer-generated images.

PyTorch implementation of Representation learning for Image-to-Image Translation. The goal of this work is to make use of
Qunatized discrete codebook to translate images from one domain to another by making a model to learn realistic discrete representation 
from the Synthetic images and utilizing this representation to genarate more realistic image. 

##### Creating environment

`conda env create -f environment.yml`


## Image Translation examples:
![Picture1](https://user-images.githubusercontent.com/71276798/225160690-155ac65c-67cc-43e4-8816-85cebb99b266.png)

![Picture2](https://user-images.githubusercontent.com/71276798/225160913-dd707469-6d7c-496f-b31d-63c482ef6862.png)

