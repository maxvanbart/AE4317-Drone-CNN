import torch

class Conv2d(object):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = torch.Tensor(out_channels,
                                   in_channels,
                                   kernel_size,
                                   kernel_size)
        self.bias = torch.Tensor(out_channels)

        # Initialize parameters
        self.init_params()

    def init_params(self, std=0.7071):

        self.weight = std * torch.randn_like(self.weight)
        self.bias = torch.rand_like(self.bias)

    def forward(self, x):
        """
            x: input tensor which has a shape of (N, C, H, W)
            y: output tensor which has a shape of (N, F, H', W') where
                H' = 1 + (H + 2 * padding - kernel_size) / stride
                W' = 1 + (W + 2 * padding - kernel_size) / stride
        """

        # Pad the input
        x_padded = torch.nn.functional.pad(x, [self.padding] * 4)

        # Unpack the needed dimensions
        N, _, H, W = x.shape

        # Calculate output height and width
        Hp = 1 + (H + 2 * self.padding - self.kernel_size) // self.stride
        Wp = 1 + (W + 2 * self.padding - self.kernel_size) // self.stride

        # Create an empty output to fill in
        y = torch.empty((N, self.out_channels, Hp, Wp), dtype=x.dtype, device=x.device)

        for i in range(Hp):
            for j in range(Wp):

                h_offset = i * self.stride
                w_offset = j * self.stride

                window = x_padded[:, :, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size]

                for k in range(N):
                    # Calculate all channel values of kth output
                    y[k, :, i, j] = (window[k] * self.weight).sum(dim=(1, 2, 3)) + self.bias


        # Cache padded input to use in backward pass
        self.cache = x_padded

        return y

    def backward(self, dupstream):
        """
            dupstream: Gradient of loss with respect to output of this layer.

            dx: Gradient of loss with respect to input of this layer.
        """

        # Unpack cache
        x_padded = self.cache

        dx_padded = torch.zeros_like(x_padded)

        self.weight_grad = torch.zeros_like(self.weight)

        # Unpack needed dimensions
        N, _, Hp, Wp = dupstream.shape

        # Loop through dupstream
        for i in range(Hp):
            for j in range(Wp):

                # Calculate offset for current window on input
                h_offset = i * self.stride
                w_offset = j * self.stride

                # Get current window of input and gradient of the input
                window = x_padded[:, :, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size]
                dwindow = dx_padded[:, :, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size]

                # Walk through each sample of the input and accumulate gradients
                # of both input and weight
                for k in range(N):
                    dwindow[k] += (self.weight * dupstream[k, :, i, j].view(-1, 1, 1, 1)).sum(dim=0)
                    self.weight_grad += window[k].view(1, self.in_channels, self.kernel_size,
                                                       self.kernel_size) * dupstream[k, :, i, j].view(-1, 1, 1, 1)
        # Calculate actual size of input height and width
        H = x_padded.shape[2] - 2 * self.padding
        W = x_padded.shape[3] - 2 * self.padding

        # Unpad dx
        dx = dx_padded[:, :, self.padding:self.padding + H, self.padding:self.padding + W]

        # Calculate bias gradients
        self.bias_grad = dupstream.sum(dim=(0, 2, 3))

        return dx