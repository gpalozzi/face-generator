# Face Generator
A DCGAN network trained on a dataset of faces to generate new images of faces that look as realistic as possible.

The network has been trained using a Generator and a Discriminator with the following hyperparameters:

## Generator
Linear fully-connected layer = 100
Convolutional dimensions = [128, 64, 32]

## Discriminator
Convolutional dimensions = [32, 64, 128]

Both models use the same Adam optimizers with the following parameters:

Learning Rate = 0.0002
Beta 1 = 0.5
Beta 2 = 0.999

# Result

These are the generated faces after 200 epochs

![](https://i.postimg.cc/1XXcQx1r/generated-faces.jpg)
