class Softmax(object):
    """
    softmax layer
    """

    def forward(self, output):
        """
        Forward pass of softpool layer
        Args:
        output: output vector with length of (tx, ty, tw, th, obj, Pa, Pb, Pc, Pd, Pe, Pf)

        Returns:
        y: output vector with shape of (Pa, Pb, Pc, Pd, Pe, Pf) where
        """

        ########################################################################
        #    Implement forward pass of softmax layer.                #
        ########################################################################

        boxes = len(output)/11
        for i in boxes:
            out_arr = output[1*boxes:11*boxes]
            probabilities = out_arr[6:11]
            total = sum(np.exp(probabilities))
            for j in probabilities:
                result = np.exp(probabilities(j))/total

        y = torch.empty(result, dtype=x.dtype, device=x.device)

import numpy as np
import torch

output = [1,2,3,4,5,1,2,3,4,5,6,1,2,3,4,5,1,2,3,4,5,6]
boxes = int(len(output)/11)
print(boxes)
# y = np.zeros_like(probabilities)
for i in range(boxes):
    print(i)
    out_arr = output[i*11:i*boxes+11]
    print(out_arr)
    dims = np.array(out_arr[:5])
    probabilities = out_arr[5:]
    nom = np.exp(probabilities)
    denom = sum(np.exp(probabilities))
    probabilities = nom/denom
    y = [dims,probabilities]
    # for j in range(len(probabilities)):
    #     print(j)
    #     y[j] = np.exp(probabilities[j]) / total
    print(y)
