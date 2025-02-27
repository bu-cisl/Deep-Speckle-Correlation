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
You can download pre-trained weights from [here](https://zenodo.org/records/14939667?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYxN2Q4ZDM0LTc3MWQtNDdiOS04MDY2LWQzYjM5MmFkZGE1YSIsImRhdGEiOnt9LCJyYW5kb20iOiI4N2I4MzY5YjNkZjRmMjMzYTAyMTFiMDI5NjQwYzk0NiJ9._LjMaI7t13wCyQA6MF4cBMacQ9SI8GrmuwaTBiIOKWfRrldPYZRJxjHKr4kvciulcubskLhg8xF_U55eEqGCnQ)

### Download dataset
You can download dataset from [here](https://zenodo.org/records/14934266?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA4NWZjOWNkLTAwNzctNGIyNi04ODNkLTIzOTIxYzA2NTg1ZCIsImRhdGEiOnt9LCJyYW5kb20iOiIwMTVlNTA0YjE2N2RjNTQ3NjlmOTQ4ZWM1MDE3MmY4NyJ9.7FuO7_HZdT-pXfJY5NHey6tZ_H4YwC1QEYfROznirjCO_OZNawN-CpaB6Brb6Qrona-rabd3NeOcQWlNAcOPwg)

### How to use
After download the pre-trained weights file, put it under the root directory and run [demo.py](demo.py).


### Results
<p align="center">
  <img src="/images/img3.png">
</p>


## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
