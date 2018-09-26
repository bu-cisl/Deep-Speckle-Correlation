# Deep-Speckle-Correlation
Python implementation of paper: **Deep speckle correlation: a deep learning approach towards scalable imaging through scattering media**. We provide model, pre-trained weights(download link available below), test data and a quick demo.


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yunzhe Li, Yujia Xue, and Lei Tian, "Deep speckle correlation: a deep learning approach toward scalable imaging through scattering media," Optica 5, 1181-1190 (2018)**](https://www.osapublishing.org/optica/abstract.cfm?uri=optica-5-10-1181)


### Abstract
Imaging through scattering is an important, yet challenging problem. Tremendous progress has been made by exploiting the deterministic input-output transmission matrix for a fixed medium. However, this one-for-one approach is highly susceptible to speckle decorrelations -- small perturbations to the scattering medium lead to model errors and severe degradation of the imaging performance. Our goal here is to develop a new framework that is highly scalable to both medium perturbations and measurement requirement.  To do so, we propose a statistical one-for-all deep learning technique that encapsulates a wide range of statistical variations for the model to be resilient to speckle decorrelations. Specifically, we develop a convolutional neural network (CNN) that is able to learn the statistical information contained in the speckle intensity patterns captured on a set of diffusers having the same macroscopic parameter. We then show for the first time, to the best of our knowledge, that the trained CNN is able to generalize and make high-quality object prediction through an entirely different set of  diffusers of the same class. Our work paves the way to a highly scalable deep learning approach for imaging through scattering media. 

<p align="center">
  <img src="/images/img1.png">
</p>


### Requirements
python 3.6

keras 2.1.2

tensorflow 1.4.0

numpy 1.14.3

h5py 2.7.1

matplotlib 2.1.2


### CNN architecture
<p align="center">
  <img src="/images/img2.png">
</p>


### Download pre-trained weights
You can download pre-trained weights from [here](https://www.dropbox.com/s/e1qcrv9o3i0h8z3/pretrained_weights.hdf5?dl=0)


### How to use
After download the pre-trained weights file, put it under the root directory and run [demo.py](demo.py).


### Results
<p align="center">
  <img src="/images/img3.png">
</p>


## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
