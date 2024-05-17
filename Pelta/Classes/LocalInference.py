import torch
import numpy as np
import os
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersSAGA
from ExperimentConfig import ExperimentConfig
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels, ResNetPytorch
import collections
from collections import OrderedDict
import json
import time
import random
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

def inferenceOnRadiator(model, device):
    imagenet_int_to_str = {}

    with open('./data/ilsvrc2012_wordnet_lemmas.txt', 'r') as f:
        for i in range(1000):
            row = f.readline()
            row = row.rstrip()
            imagenet_int_to_str.update({i: row})

    # Show the MAX_PREDS highest scoring labels:
    MAX_PREDS = 5
    # Do not show labels with lower score than this:
    MIN_SCORE = 0.8 
    import matplotlib.pyplot as plt
    def show_preds(logits, image, correct_flowers_label=None, tf_flowers_logits=False):

        if len(logits.shape) > 1:
            logits = torch.reshape(logits, [-1])

        fig, axes = plt.subplots(1, 2, figsize=(7, 4), squeeze=False)

        ax1, ax2 = axes[0]

        ax1.axis('off')
        print("image.cpu().numpy size ", image.cpu().numpy().shape)
        matr = image.cpu().numpy()
        ax1.imshow(np.moveaxis(matr,0, -1))
        classes = []
        scores = []
        # print("logits ", logits)
        print("[L] logits shape ", logits.shape)
        logits = logits.detach().cpu().numpy()
        logits_max = np.max(logits)
        softmax_denominator = np.sum(np.exp(logits - logits_max))
        for index, j in enumerate(np.argsort(logits)[-MAX_PREDS::][::-1]):
            score = 1.0/(1.0 + np.exp(-logits[j]))
            if score < MIN_SCORE: break
            # predicting in imagenet label space
            classes.append(imagenet_int_to_str[j])
            print("[?] j ", j)
            print("[?] imagenet_int_to_str[j] ", imagenet_int_to_str[j])
            scores.append(np.exp(logits[j] - logits_max)/softmax_denominator*100)

        ax2.barh(np.arange(len(scores)) + 0.1, scores)
        ax2.set_xlim(0, 100)
        ax2.set_yticks(np.arange(len(scores)))
        ax2.yaxis.set_ticks_position('right')
        ax2.set_yticklabels(classes, rotation=0, fontsize=14)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.set_xlabel('Prediction probabilities', fontsize=11)
        plt.savefig("JUSTATRY")

    from PIL import Image
    image = Image.open("./data/ILSVRC2012_val_00013683.JPEG")
    from torchvision.transforms import ToTensor
    image = ToTensor()(image).unsqueeze(0)
    import torchvision.transforms as transforms
    transformTest = transforms.Compose([
        transforms.Resize((300, 300)),
    ])
    print("[?] type(image) ", type(image))
    print("[?] image.size() ", image.size())
    # image = transformTest(image).to(device)
    image = transformTest(image).to(device)
    print("[?] image.size() ", image.size())
    print("[?] type(image) ", type(image))
    print("[?] image.dtype ", image.dtype)
    model.to(device)
    # Run model on image
    logits = model(image)

    # print(logits)
    # print(logits.shape)

    # Show image and predictions
    show_preds(logits, image[0])