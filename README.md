# Deep-Speckle-Correlation
Python implementation of paper: **Deep speckle correlation: a deep learning approach towards scalable imaging through scattering media**. We provide model, pre-trained weights, test data and a quick demo.



### Citation
If you find this project useful in your research, please consider citing our paper:
[**Li, Y., Xue, Y. and Tian, L., 2018. Deep speckle correlation: a deep learning approach towards scalable imaging through scattering media. arXiv preprint arXiv:1806.04139.**](https://arxiv.org/abs/1806.04139)



### Abstract
Imaging through scattering is an important, yet challenging problem. Tremendous progress has been made by exploiting the deterministic input-output transmission matrix for a fixed medium. However, this one-for-one approach is highly susceptible to speckle decorrelations -- small perturbations to the scattering medium lead to model errors and severe degradation of the imaging performance. Our goal here is to develop a new framework that is highly scalable to both medium perturbations and measurement requirement.  To do so, we propose a statistical one-for-all deep learning technique that encapsulates a wide range of statistical variations for the model to be resilient to speckle decorrelations. Specifically, we develop a convolutional neural network (CNN) that is able to learn the statistical information contained in the speckle intensity patterns captured on a set of diffusers having the same macroscopic parameter. We then show for the first time, to the best of our knowledge, that the trained CNN is able to generalize and make high-quality object prediction through an entirely different set of  diffusers of the same class. Our work paves the way to a highly scalable deep learning approach for imaging through scattering media. 
![Alt Text](/images/img1.png)



### Requirements
Python 3.6

Keras 2.1.2

Tensorflow 1.4.0

Numpy 1.14.3

H5py 2.7.1

Matplotlib 2.1.2



### Download pre-trained weights
You can download pre-trained weights from [here](https://www.dropbox.com/s/e1qcrv9o3i0h8z3/pretrained_weights.hdf5?dl=0)



### CNN architecture
![Alt Text](/images/img2.png)



### How to use
Please refer to [demo.py](demo.py).



### Results
![Alt Text](/images/img3.png)



## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
