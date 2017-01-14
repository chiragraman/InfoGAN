# Multicomp Hackathon January 2017 - GAN and InfoGAN

This repository contains code for experiments carried out for the CMU Multicomp Hackathon, intended to be an excercise in 
exploring new topics of interest. This project involves an exploratory study of Generative Adversarial Networks, and the 
variation InfoGAN.

## InfoGAN

Code for reproducing key results in the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

### Dependencies

Install tensorflow following the instructions from the [tensorflow repository](https://github.com/tensorflow/tensorflow).
We recommend installing all the dependencies in a separate virtual environment.

In addition, the following packages are required:
- `prettytensor`
- `progressbar`
- `python-dateutil`

The easiest way to install all dependencies is running the following command:
`pip install -r requirements.txt`

The code has been tested with tensorflow version 0.12.1 and prettytensor version 0.6.2, as mentioned in the file `requirements.txt`.

### Running in Docker

```bash
$ git clone git@github.com:openai/InfoGAN.git
$ docker run -v $(pwd)/InfoGAN:/InfoGAN -w /InfoGAN -it -p 8888:8888 gcr.io/tensorflow/tensorflow:r0.9rc0-devel
root@X:/InfoGAN# pip install -r requirements.txt
root@X:/InfoGAN# python launchers/run_mnist_exp.py
```

### Running Experiment

We provide the source code to run the MNIST example:

```bash
PYTHONPATH='.' python launchers/run_mnist_exp.py
```

For executing on GPU's, use the following command:

```bash
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=<device number> python launchers/run_mnist_exp.py
```
where `<device number>` refers to the GPU device to be used; for instance, a device number of 0 would pick the first GPU 
registered.

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/mnist
```

## GAN

Code for reproducing key results in the paper [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf) by Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.

### Dependencies

- Keras 1.2.0
- [Theano](http://deeplearning.net/software/theano/)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) tqdm, matplotlib

