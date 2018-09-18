---
layout: default
---
This is a preliminary report about the tutorial on Variational AutoEncoder. This report consists of short summary about the concepts and experiments that will be discussed in the tutorial.

# Outline
- Basics of autoencoders
- Generative models
- Intuition behind Variational autoencoder
- Maths behind Variational autoencoder
- Experiments 
  - Latent Space Visualizations
  - Visualization of cluster formation
  - Effect of change in weightage for KL divergance during training
  - Effect of weightage of KL divergance on disentangled representation learning
  - Shortcoming of VAE
- Applications of VAE

## Autoencoders

These are machine learning models under unsupervised learning that come with a goal to learn good representations by trying to reconstruct the input itself. Main problem of autoencoders is not let it learn a identity function which is alleviated by regularized autoencoders(eg Sparse Autoencoders). Sparse Autoencoders come with a motive of getting sparse representations in latent space which essentially means that only few neurons are active for a particular data point. This sparse constraint in latent space forces the model to learn more good representations. In regularized autoencoders, we actually misuse the meaning of regularization. By definition regularization is our prior belief on distribution of model’s parameters where as in regularized autoencoders the regularization is a prior assumed on latent space which is **not on parameters rather on data.**

## Generative Models

These type of machine learning models come with a goal to learn the true data distribution. Intuitive motivation is that if a model is able to generate plausible samples close to train data distribution then it must have learnt very well representations too. These are useful where data collection is hard or next to impossible, henceforth we could use generative models to generate samples in order to augment in existing dataset.

## Intuition behind VAE

Before jumping to VAE, we would like to first connect the autoencoders with probabilistic graphical models. We can view the encoder as approximating conditional probability distribution -- p(z\|x) where z is denoting latent space random var & x is input data point. Similarly decoder can be viewed as approximating a conditional distribution -- q(x\|z). So we would like to use the decoder part as generative model since it learns to map a point in latent space to a point in input space. For a decoder to get trained we also want encoder(only for training) as it provides a meaning full point in latent space corresponding to an input. But issue is that both our encoders & decoders are deterministic functions but a generative model should be stochastic (Eg : stochasticity can be seen as generating images of 3 digit but in different orientations). So to make our traditional Autoencoders stochastic we make use of re-parametrization trick which serves as core of VAE learning.

## Maths behind VAE

On observing final loss function we get two terms i.e KL term & likelihood term. KL term helps to restrict encoder’s learnt latent space distribution as close as possible to our prior. Likelihood term helps the decoder to reconstruct the images.

## Experiments

The main aim of performing experiments is to find some insights that would be helpful in understanding the intuition behind VAE. This experiments would be mainly performed on MNIST and if required some will also be performed on CIFAR10 or fashion MNIST

#### The Experiments are as follows:

* Visualization of the latent space representations
> Visualizing the effect of KL Divergence on the latent space representations. T-SNE will used in order to obtain the visualization of the representation obtained by using KL Divergence and without using KL Divergence. Both the visualizations will be compared in-order to obtain useful statistics and analysis
* Visualization of the cluster formation 
> T-SNE visualization of the representation of latent space in Normal AutoEncoder and Variational AutoEncoder will be compared in order to understand the difference in the cluster formation and the reason behind it.
* Generation of blurry samples
> One of the disadvantages of VAE is that it leads to the generation of blurry samples. we except to demonstrate that effect through this experiment.
* Experiment fact on learning VAE
> once we let increase the KL term in loss function then learning is more stable. Intuitively it means once we set a loose bound on learnt distribution to match our prior, we get decoder very fast learnt and then KL term comes in picture.

## Applications of VAE

#### Disentangled representation using variational autoencoder

Variational autoencoder can also be used for learning disentangled representations. Disentangled representations are the representations in which the individual output of the neurons in the latent space are uncorrelated that is each neuron in the latent space represents some unique feature present in the input. In order to implement this class of variational autoencoder is to only add an extra hyperparameter named Beta which will act a weight on the KL divergence term of the loss function. Thus if we enforce the KL divergence term with a very high weight then this would force the network to have efficient compression of information in the latent code leading to disentangled representation.

#### Denoising Autoencoders

Autoencoders are neural networks which are commonly used for feature extraction or compression. However, if we have same or more number of nodes in the latent space then this will lead the autoencoder to not learn anything useful but to just pass the input as it is through the layers. But if we pass the input with some amount of noise and try to reconstruct the input but without the noise then this extra nodes in the latent space will learn to remove the noise from the input leading to an autoencoder which can be used for denoising the input. Therefore, this type of autoencoder is known as denoising autoencoder.

