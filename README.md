# Fooling Computer Vision in Autonomous Vehicles

This project have been developed for a Computer Science course on Cybersecurity from the engineering school CentraleSupélec. 

# DeepFool
This tool is based on the Github Project [DeepFool](https://github.com/LTS4/DeepFool).  
We use it to modify images, making them mis-classified by the Deep Learning model ResNet34.
These modification are invisible to the human eye.

## Installation
Ensure that you have python3 and pip, then run:
```bash
pip3 install -r requirements.txt

```
Note : This command will install Torch, ~500MB  

You will also need tkinter
```bash
sudo apt-get install python-tk python3-tk
```

## Functionment description

### deepfool.py

This function implements the algorithm proposed in [[1]](http://arxiv.org/pdf/1511.04599) using PyTorch to find adversarial perturbations.

The parameters of the function are:

- `image`: Image of size `HxWx3d`
- `net`: neural network (input: images, output: values of activation **BEFORE** softmax).
- `num_classes`: limits the number of classes to test against, by default = 10.
- `max_iter`: max number of iterations, by default = 50.

### test_deepfool.py

Computes the adversarial perturbation using SSI team addings, like :
- Shadow methods: adversarial labels skipped to go deeper in Fooling
- Rerun the fooling loop

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.


# Lisa CNN attack
We used this project to compute where to put stickers on a stop panel and fool Deep Learning models.  
We used the [RP2 Project](https://github.com/evtimovi/robust_physical_perturbations/tree/master/lisa-cnn-attack)
## Installation
With pipenv installed, run
```bash
cd lisa-cnn-attack
rm Pipfile
pipenv install tensorflow==1.4.1
pipenv install keras==1.2.0
pipenv install scipy
pipenv install opencv-python
pipenv install pillow
pipenv shell
```
This process can be quite long depending your internet connection.

## Proceed to the Machine Duping
To run the attack: 
```bash
sh run_attack_many.sh
```
It will create several perturbed images in the optimization_output/l1basedmask_uniformrectangles/noisy_images folder.  
Take one of the images (the last one for instance) and put it in a new folder, ‘test_set’.  
You can try to classify the perturbed image : 
```bash
python manyclassify.py --attack_srcdir optimization_output/l1basedmask_uniformrectangles/test_set
```
