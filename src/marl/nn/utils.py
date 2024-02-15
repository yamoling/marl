import torch


def make_cnn(input_shape: tuple[int, int, int], filters: list[int], kernel_sizes: list[int], strides: list[int]):
    """Create a CNN with flattened output based on the given filters, kernel sizes and strides."""
    channels, height, width = input_shape
    output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides)
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    print(f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}")
    modules = []
    for f, k, s in zip(filters, kernel_sizes, strides):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    output_size = output_h * output_w * filters[-1]
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width, input_height, kernel_sizes, strides) -> tuple[int, int]:
    """Compute the output width and height of a sequence of 2D convolutions"""
    width = input_width
    height = input_height
    for kernel_size, stride in zip(kernel_sizes, strides):
        width = (width - (kernel_size - 1) - 1) // stride + 1
        height = (height - (kernel_size - 1) - 1) // stride + 1
    return width, height
