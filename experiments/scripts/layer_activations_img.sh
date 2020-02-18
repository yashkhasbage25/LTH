#!/bin/sh

python layer_activations_img.py -m VGG16_bn -d SVHN -r "model=VGG16_bn,dataset=SVHN,run1" -cuda 3 -ckpts "final_weights_Pm_1.0000014305114746.pth" "final_weights_Pm_21.544349670410156.pth" "final_weights_Pm_35.938140869140625.pth" "final_weights_Pm_100.0.pth"