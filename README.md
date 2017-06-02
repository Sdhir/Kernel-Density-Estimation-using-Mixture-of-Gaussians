# Kernel-Density-Estimation-using-Mixture-of-Gaussians

The mean of log-likelihood is computed on given datasets using kernel density
estimation based on mixture of Gaussians distributions. The standard devia-
tion () corresponding to a Gaussian component is considered optimal when
there is a peak value observed in the computation of log-likelihood. This work
is a maximum likelihood estimation on given datasets. The realized model has
successfully found out the optimal standard deviation of Gaussian kernel on
the given MNIST and CIFAR100 datasets.

The kernel density estimation with mixture of Gaussians is one of the
simplest model to construct a probabilistic model, but in general the model is
said to be biased when the data is bounded.

###### CIFAR - 100
![alt text](https://github.com/Sdhir/Kernel-Density-Estimation-using-Mixture-of-Gaussians/blob/master/cifar.png)
###### MNIST
![alt text](https://github.com/Sdhir/Kernel-Density-Estimation-using-Mixture-of-Gaussians/blob/master/mnist.png)
