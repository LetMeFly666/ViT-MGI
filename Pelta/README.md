# Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments
Code base for the **Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments** paper (Queyrut, Schiavoni & Felber) accepted at ICDCS'23 (open access version soon available).

Code is provided for applying the Pelta defense scheme to an ensemble of Vision Transformer (ViT-L-16) and and Big Transfer Model (BiT-M-R101x3) against the Self-Attention Gradient Attack (original attack code from authors, [paper here](https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html)). The defense provided here works for CIFAR-10 and was coded entirely on PyTorch.
Parameters of the defense can be changed in the `env` file through the `PELTA` and `SHIELDED`parameters (set to `True` and `BOTH` by default).

# Step by Step Guide

<ol>
  <li>Install the packages listed in the Software Installation Section (see below).</li>
  <li>Download the models from this Kaggle <a href="www.kaggle.com/reyacardov/ensemblemodels">dataset link</a>
  <li>Move both models into the ".\ExtendedPelta\Models" folder</li>
  <li>Run the main in the Python IDE of your choice</li>
</ol>

# Software Installation 

We use the following software packages: 
<ul>
  <li>pytorch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>numpy==1.19.2</li>
  <li>opencv-python==4.5.1.48</li>
  <li>python-dotenv==0.21.1</li>
</ul>



# System Requirements 

All our defenses were run on one 40GB A100 GPU and system RAM were of 16GB.
