class Softmax(object):
    """
    softmax layer
    """

    def __init__(self, ):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, output):
        """
        Forward pass of softpool layer
        Args:
        output: output vector with length of (tx, ty, tw, th, obj, Pa, Pb, Pc, Pd, Pe, Pf)

        Returns:
        y: output vector with shape of (N, C, H', W') where
        H' = 1 + (H + 2 * padding - kernel_size) / stride
        W' = 1 + (W + 2 * padding - kernel_size) / stride
        """


        # Unpack the needed dimensions
        N, C, H, W = x.shape
        KS = self.kernel_size

        y = torch.empty((N * C, Hp, Wp), dtype=x.dtype, device=x.device)

        ########################################################################
        #    Implement forward pass of 2D avg pooling layer. It is very   #
        #   similar to convolutional layer that you implemented before. Again  #
        #                  have a look at the backward pass.                   #
        ########################################################################

        

        # Loop through each of the output value
        for i in range(Hp):
            for j in range(Wp):
                # Calculate offsets on the input
                h_offset = i * self.stride
                w_offset = j * self.stride

                # Get the corresponding window of the input
                window = x_padded[:, :, h_offset:h_offset + KS, w_offset:w_offset + KS].reshape(N * C, -1)

                y[:, i, j], _ = window.mean(dim=1)