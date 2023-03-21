class MaxPool2d(object):
    """
    2D max pooling layer
    """

    def __init__(self, kernel_size, stride=1, padding=0):
        """
        Initialize the layer with given parameters:
        Args:
        kernel_size: height and width of the receptive field of the layer in pixels.
        stride: # pixels between adjacent receptive fields in both horizontal and vertical directions.
        padding: # pixels that is used to zero-pad the input.
        """

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Forward pass of max pooling layer
        Args:
        x: input tensor with shape of (N, C, H, W)

        Returns:
        y: output tensor with shape of (N, C, H', W') where
        H' = 1 + (H + 2 * padding - kernel_size) / stride
        W' = 1 + (W + 2 * padding - kernel_size) / stride
        """

        # Pad the input
        x_padded = torch.nn.functional.pad(x, [self.padding] * 4)

        # Unpack the needed dimensions
        N, C, H, W = x.shape
        KS = self.kernel_size

        # Calculate output height and width
        Hp = 1 + (H + 2 * self.padding - KS) // self.stride
        Wp = 1 + (W + 2 * self.padding - KS) // self.stride

        # Create an empty output to fill in.
        # We combine first and second dim to speed up as we need no loop for each
        # channel.
        y = torch.empty((N * C, Hp, Wp), dtype=x.dtype, device=x.device)

        ########################################################################
        #    Implement forward pass of 2D max pooling layer. It is very   #
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

                y[:, i, j], _ = window.max(dim=1)