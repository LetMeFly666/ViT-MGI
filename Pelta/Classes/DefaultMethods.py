#In this file we provide different methods to run attacks on different models 
import torch
import numpy as np
import os
import ShuffleDefense
from ModelPlus import ModelPlus
import DataManagerPytorch as DMP
import AttackWrappersSAGA
from ExperimentConfig import ExperimentConfig
from TransformerModels import VisionTransformer, CONFIGS
import BigTransferModels, ResNetPytorch, LocalInference
import collections
from collections import OrderedDict
import json
import time
import random
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

load_dotenv('/content/pelta_extended/ExtendedPelta/env')


# From model name creates model, loads checkpoint, pours into model then creates
# ModelPlus and returns it
def SAGA_model_plus(model_name, device, numClasses, imgSize, batchSize, vis=False):
    if model_name in ("ViT-L-16", "ViT-L/16", "ViT-L_16"):
        config = CONFIGS["ViT-L_16"]
        # Load clear model (will be tested against adversarial examples generated
        # against shielded model)
        model = VisionTransformer(config, imgSize, zero_head=True, num_classes=numClasses, vis=vis)
        MODEL_VIT_DIR= os.getenv('MODEL_VIT_DIR')
        checkptdict = np.load(MODEL_VIT_DIR) if 'npz' in MODEL_VIT_DIR else torch.load(MODEL_VIT_DIR)
        MODEL_VIT_DIR= os.getenv('MODEL_VIT_DIR')
        if 'npz' in MODEL_VIT_DIR:
            checkptdict = OrderedDict(zip(("{}".format(k) for k in checkptdict), (checkptdict[k] for k in checkptdict)))
            model.load_from(np.load(MODEL_VIT_DIR))
        else:
            model.load_state_dict(checkptdict)
        model.eval()

        # Wrap the model in the ModelPlus class
        
        modelPlusV = ModelPlus("ViT-L_16", model, device, imgSizeH=imgSize, imgSizeW=imgSize,
                            batchSize=batchSize)
        return modelPlusV
    if model_name in ("BiT-M-R101x3", "BiT-M_R101x3", "BiT_M-R101x3", "BiT_M_R101x3"):
        MODEL_CNN_DIR = os.getenv('MODEL_CNN_DIR')
        model = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=numClasses, zero_head=False)
        MODEL_CNN_DIR= os.getenv('MODEL_CNN_DIR')
        #Get the checkpoint 
        
        checkptdict = np.load(MODEL_CNN_DIR) if 'npz' in MODEL_CNN_DIR else torch.load(MODEL_CNN_DIR)
        if 'npz' in MODEL_CNN_DIR:
            checkptdict = OrderedDict(zip(("{}".format(k) for k in checkptdict), (checkptdict[k] for k in checkptdict)))
            if not (os.getenv("PELTA")=="True" and os.getenv("SHIELDED") in ("CNN", "BOTH")):
                model.load_from(checkptdict)

            if os.getenv("PELTA")=="True" and os.getenv("SHIELDED") in ("CNN", "BOTH"):
                new_state_dict = OrderedDict()
                for k, v in checkptdict.items():
                    name = k
                    if name != "resnet/root_block/standardized_conv2d/kernel":
                        new_state_dict[name] = v
                    else:
                        np.save(f'rootWeights{os.getenv("DATASET")}.npy', v)
                checkptdict = new_state_dict
                #Load the dictionary
                model.load_from(checkptdict)
        else:
            if os.getenv("DATASET") in ("CIFAR10", "CIFAR100"):
                # We actually have checkptdict.keys()=dict_keys(['step', 'model', 'optim']) so:
                checkptdict = checkptdict['model']
                # Also checkptdict keys are module.root.conv... instead of simply root.conv
                # so we remove the "module" through the next line:
                checkptdict = OrderedDict(zip(("{}".format(k[7:]) for k in checkptdict), (checkptdict[k] for k in checkptdict)))
            # If Pelta is usd and we're shielding BiT, then we don't enter this condition. Instead...
            if not (os.getenv("PELTA")=="True" and os.getenv("SHIELDED") in ("CNN", "BOTH")):
                model.load_state_dict(checkptdict)
        # ...we enter this one.
        # If Pelta is used, the weights have to be saved to allow for weights loading in special layers in the respective model classes
        if os.getenv("PELTA")=="True" and os.getenv("SHIELDED") in ("CNN", "BOTH") and not ('npz' in MODEL_CNN_DIR):
            # print("[?] checkptdict.items() ", checkptdict.items())
            new_state_dict = OrderedDict()
            for k, v in checkptdict.items():
                name = k
                if name != "root.conv.weight":
                    new_state_dict[name] = v
                else:
                    np.save(f'rootWeights{os.getenv("DATASET")}.npy', v.cpu().detach().numpy())
            checkptdict = new_state_dict
            #Load the dictionary
            model.load_state_dict(checkptdict)

        # We want checkpoint to be an OrderedDict to be parsed if we need Pelta 
        if not isinstance(checkptdict, collections.OrderedDict):
            checkptdict = checkptdict["model"]

        model.eval()

        if os.getenv('DATASET')=='IMAGENET':
            LocalInference.inferenceOnRadiator(model, device)

        # Wrap the model in the ModelPlus class
        # -------------------------------------
        #Here we hard code the Big Transfer Model Plus class input size to 160x128 
        #(what it was trained on)
        modelBig101Plus = ModelPlus("BiT-M-R101x3", model, device,
                                    imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)

        return modelBig101Plus
    if model_name in ("BiT-M-R152x4", "BiT-M_R152x4", "BiT_M-R152x4", "BiT_M_R152x4"):
        MODEL_CNN_DIR = os.getenv('MODEL_CNN_DIR')
        model = BigTransferModels.KNOWN_MODELS["BiT-M-R152x4"](head_size=numClasses, zero_head=False)
        MODEL_CNN_DIR= os.getenv('MODEL_CNN_DIR')
        #Get the checkpoint 
        checkptdict = np.load(MODEL_CNN_DIR) if 'npz' in MODEL_CNN_DIR else torch.load(MODEL_CNN_DIR)
        if 'npz' in MODEL_CNN_DIR:
            checkptdict = OrderedDict(zip(("{}".format(k) for k in checkptdict), (checkptdict[k] for k in checkptdict)))
            model.load_from(checkptdict)
        else:
            model.load(checkptdict)
        if os.getenv("PELTA")==True:
            #Remove module so that it will load properly
            new_state_dict = OrderedDict()
            # We want checkpoint to be an OrderedDict to be parsed if we need Pelta 
            if not isinstance(checkptdict, collections.OrderedDict):
                checkptdict = checkptdict["model"]

            for k, v in checkptdict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v # comment this...
                ## ...and uncomment the "if" + add the rootWeights.npy to
                ## make BiT shield work.
                # --------
                # if name != "root.conv.weight":
                #     new_state_dict[name] = v
                # else:
                #     np.save('rootWeights.npy', v.cpu().detach().numpy())
            checkptdict = new_state_dict
            #Load the dictionary
            model.load_state_dict(checkptdict)

        # We want checkpoint to be an OrderedDict to be parsed if we need Pelta 
        if not isinstance(checkptdict, collections.OrderedDict):
            checkptdict = checkptdict["model"]

        model.eval()

        LocalInference.inferenceOnRadiator(model, device)

        # Wrap the model in the ModelPlus class
        # -------------------------------------
        #Here we hard code the Big Transfer Model Plus class input size to 160x128 
        #(what it was trained on)
        modelBig101Plus = ModelPlus("BiT-M-R152x4", model, device,
                                    imgSizeH=imgSize, imgSizeW=imgSize, batchSize=batchSize)
        return modelBig101Plus
        return modelBig101Plus
    if model_name == "ResNet-164":
        #Load the ResNet-164
        dirR = "Models/ModelResNet164-Run0.th"

        ##########
        # WORKING 
        ##########
        modelR = ResNetPytorch.resnet164(inputImageSize=32, numClasses=numClasses)
        #Get the checkpoint 
        checkpoint = torch.load(dirR, map_location="cpu")
        #Load the dictionary
        modelR.load_state_dict(checkpoint['state_dict'])
        modelR.eval()

        # Wrap the model in the ModelPlus class
        modelRes164Plus = ModelPlus(model_name, modelR, device,
                                    imgSizeH=32, imgSizeW=32, batchSize=batchSize)
        return modelRes164Plus

#Validate using a dataloader 
def validateBoth(valLoader, modelPlusList, device=None):
    #switch to evaluate mode
    modelPlusList[0].eval()
    modelPlusList[1].eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            
            output_ViT = modelPlusList[0](inputVar)
            output_ViT = output_ViT.float()

            output_BiT = modelPlusList[1](inputVar)
            output_BiT = output_BiT.float()

            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output_ViT[j].argmax(axis=0) == target[j] and output_BiT[j].argmax(axis=0):
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Load the shuffle defense containing ViT-L-16 and BiT-M-R101x3
#For all attacks except SAGA, vis should be false (makes the Vision tranformer
#return the attention weights if true)
def LoadShuffleDefenseAndCIFAR10(vis=False):
    #Basic variable and data setup
    device = torch.device("cuda")

    NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
    IMG_SIZE = int(os.getenv("IMG_SIZE"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

    DATASET = os.getenv('DATASET')

    if DATASET == "CIFAR10":
        valLoader = DMP.GetCIFAR10Validation(IMG_SIZE, 8)
    elif DATASET == "CIFAR100":
        valLoader = DMP.GetCIFAR100Validation(IMG_SIZE, BATCH_SIZE)
    elif DATASET == "IMAGENET":
        valLoader = DMP.GetImageNetValidation(IMG_SIZE, BATCH_SIZE)

    # Which model pair is used for SAGA?
    vitName = os.getenv('MODEL_VIT')
    cnnName = os.getenv('MODEL_CNN')

    # Create and add the models
    modelPlusList = []           # v--'ViT-L/16'
    modelPlusVit = SAGA_model_plus(vitName, device, NUM_CLASSES, IMG_SIZE, BATCH_SIZE, vis)
    modelPlusCnn = SAGA_model_plus(cnnName, device, NUM_CLASSES, IMG_SIZE, BATCH_SIZE)

    # print("[?] modelPlusVit", modelPlusVit)
    # print("[?] modelPlusCnn", modelPlusCnn)
    modelPlusList.append(modelPlusVit)
    modelPlusList.append(modelPlusCnn)
    
    #Now time to build the defense 
    defense = ShuffleDefense.ShuffleDefense(modelPlusList, NUM_CLASSES)
    return valLoader, defense


#Run the Self-Attention Gradient Attack (SAGA) on ViT-L and BiT-M-R101x3
def SelfAttentionGradientAttackCIFAR10(): # TODO separate attack class for C100?@
    start = time.time()
    assert(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("Running Self-Attention Gradient Attack on a blend of ViT-L-16 and BiT-M-R101x3") 
    #Set up the parameters for the attack 
    attackSampleNum = int(os.getenv("ATTACK_SAMPLES_NUM"))
    numClasses = int(os.getenv("NUM_CLASSES"))
    coefficientArray = torch.zeros(2)
    secondcoeff = float(os.getenv("SECOND_COEFF"))
    coefficientArray[0] = 1.0 - secondcoeff
    coefficientArray[1] = secondcoeff
    device = torch.device("cuda")
    epsMax = 0.031
    clipMin = 0.0
    clipMax = 1.0
    numSteps = int(os.getenv("NUM_STEPS"))
    #Load the models and the dataset
    #Note it is important to set vis to true so the transformer's model output returns the attention weights 
    valLoader, defense = LoadShuffleDefenseAndCIFAR10(vis=True)
    modelPlusList = defense.modelPlusList
    #Note that the batch size will effect how the gradient is computed in PyTorch
    #Here we use batch size 8 for ViT-L and batch size 2 for BiT-M.
    #Other batch sizes are possible but they will not generate the same result
    modelPlusList[0].batchSize = 8
    modelPlusList[1].batchSize = 2
    #Get the clean examples 
    cleanLoader =AttackWrappersSAGA.GetFirstCorrectlyOverlappingSamplesBalanced(device, attackSampleNum,
     numClasses, valLoader, modelPlusList)
    #Do the attack
    advLoader = AttackWrappersSAGA.SelfAttentionGradientAttack(device, epsMax, numSteps,
     modelPlusList, coefficientArray, cleanLoader, clipMin, clipMax)
    #Go through and check the robust accuray of each model on the adversarial examples 
    predictionOfBothModels = {}
    for i in range(0, len(modelPlusList)):
        acc, prediction_list = modelPlusList[i].validateD(advLoader)
        predictionOfBothModels[modelPlusList[i].modelName]=prediction_list
        print(f"{modelPlusList[i].modelName} robust Acc: {acc}")

    adv_ex = 0
    correct_pred = 0
    misclass = 0
    # BiT_list = predictionOfBothModels["BiT-M-R101x3"]
    cnnList = predictionOfBothModels[os.getenv("MODEL_CNN")]
    vitList = predictionOfBothModels[os.getenv("MODEL_VIT")]
    for i, pred in enumerate(cnnList):
        if pred=='incorrect' and vitList[i]=='incorrect':
            adv_ex+=1
        elif pred=='incorrect' and vitList[i]=='correct !':
            misclass+=1
        elif pred=='correct !' and vitList[i]=='incorrect':
            misclass+=1
        elif pred=='correct !' and vitList[i]=='correct !':
            correct_pred+=1
        else:
            pass
            # print((pred, vitList[i]))

    blend_acc = .0
    for i, pred in enumerate(cnnList):
        if random.choice([vitList[i], pred])=='correct !':
            blend_acc+=1
    blend_acc/=len(cnnList)
    print("blend robust acc: ", blend_acc)
    modelPlusList[0].clearModel()
    modelPlusList[1].clearModel()
    end = time.time()
    print("Elapsed time: ", end - start)

