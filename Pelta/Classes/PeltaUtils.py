# In this module wew provide algorithms that simulate shielding attack/defense
# We only have 20Mb at most of available TEE - only 5e6 parameters can be shielded.
from tqdm import tqdm
import numpy as np
import torch


def kMaxIndexes(array: np.ndarray, k: int) -> list: 
    """Computes indexes of the k largest abs values from an array.
    """
    # Get abs values
    absArray = np.abs(array).flatten()  # (4096, 1024)
    # Get indexes of the k largest values from the flatten
    flatIndexesOfKMax = (-absArray).argsort()[:k]  # (k,) 
    # Convert to list of index tuples (m,n)
    return [divmod(index, array.shape[1]) for index in list(flatIndexesOfKMax)]

def shield(layerArray: torch.Tensor, teeSize: int=20e6, replacingStrategy: str="latentSpaceAverage") -> torch.Tensor:
    """Replaces some parameters of input tensor with attacker values.
    """
    shieldedArray = layerArray.cpu().detach().numpy()
    if replacingStrategy=="latentSpaceAverage":
        print(shieldedArray.shape)
        # Shatter along latent space axis
        replacingValues = np.mean(shieldedArray, axis=1) # should have 4096 values
    else:
        # TODO
        return
    # Len of the flattened
    N_parameters = shieldedArray.shape[0]*shieldedArray.shape[1]
    # How many floats can we shield at most in the TEE?
    N_maxShieldableFloats = min([int(teeSize/4), N_parameters]) # 4194304
    #                                        ^--float size
    print("[+] Shielding layer...")
    for k, tupl in enumerate(tqdm(kMaxIndexes(shieldedArray, N_maxShieldableFloats))):
        #                         ^-- index tuples of the k maximums
        shieldedArray[tupl[0]][tupl[1]] = replacingValues[tupl[0]]
    print("[+] Done.")
    return torch.from_numpy(shieldedArray)
