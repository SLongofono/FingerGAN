# FingerGAN

This is a generative adversarial network created for ECE2195, Information
Security, at the University of Pittsburgh.  Working with a modified version of
the Sokoto Coventry Fingerprint Dataset (See Usage.pdf for details).

The premise of the project is to create a neural network which can regularly
generate fake fingerprints which can fool a neural network designed to
classify fingerprints as a means of biometric authentication.

The code was modified from [this example tutorial on the PyTorch website.](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
The SOCOF dataset was modified to fit the fixed input size (64x64 pixel RGB
images).


# Installing Prerequisites
The software depends on a number of Python modules to do its job, centered
around the PyTorch suite.  Besides Python 3, a few prerequisites are required,
installed via pip

```shell
pip3 install numpy matplotlib
pip3 install torch torchvision
```

# Training
Training the dataset takes a few hours on a laptop, as we did not have a GPU
at our disposal to exploit.  The original code includes additional setup code
to offload training to a GPU.

In the data folder, two folders store the preprocessed training data.  The
positive directory contains all the left-hand fingerprints of the SOCOF
dataset, and was used to represent known fingerprints.  The negative directory
contains all the right-hand fingerprints, and may also be used to train.

To train a network and output intermediate images, run main.py:

```shell
python3 main.py
```

Every 10 generations, a sample of the generated fingerprints is saved to the
working directory.  On completion of training (25 epochs), a plot of loss
versus generation will be presented, and the internal states of the networks
will be saved to the working directory.

# Demonstration
The demo.py program creates a new discriminator network representing the
biometric authorization system which is trained on the original dataset.
Then, a new generator network is created which uses the weights and paramters
captured during the training process.  After setup, the generator will produce
new fingerprints from a random input vector, display them, and display the
output from the discriminator for that fingerprint.

The demo discriminator only uses the original dataset for training,
representing a group of known fingerprints for authorized users.  The demo
generator takes as input a random vector and produces several batches of new
fingerprints, which are then checked against the discriminator.

After running the demo program, any generated images which had confidence >
90% will be saved to the working directory for inspection.  Run the demo after
training:

```shell
python3 demo.py
```

