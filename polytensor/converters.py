import torch


def stringToUint(bitstring_tensor: "torch.Tensor", basis: int = 2):
    """
    Convert a bitstring tensor to an unsigned integer.

    Args:
        bitstring_tensor: torch.Tensor
            A tensor of bits

    Returns:
        uint_value: int
            The unsigned integer corresponding to the bitstring
    """

    # Reverse the tensor to align with the order of bits in binary representation
    reversed_tensor = torch.flip(bitstring_tensor, [-1])

    # Create a tensor of powers of the basis
    powers = torch.pow(
        basis * torch.ones_like(reversed_tensor), torch.arange(reversed_tensor.size(-1))
    )

    # Perform element-wise multiplication and sum to get the integer value
    uint_value = torch.sum(reversed_tensor * powers, dim=-1).int().unsqueeze(-1)

    return uint_value


def uintToString(uint_tensor: "torch.Tensor", num_bits: int, basis: int = 2):
    """
        Convert a batch of unsigned integers to bitstring tensors for any given basis.

    Args:
        uint_tensor: torch.Tensor
            A 1D tensor of unsigned integers to be converted.
        num_bits: int
            The number of bits/digits in each output tensor.
        basis: int
            The base for the conversion.

    Returns:
        bitstring_tensors: torch.Tensor
            A 2D tensor where each row is the bitstring representation of each integer in the batch.
    """

    # Prepare powers of the basis
    powers = torch.pow(basis * torch.ones(num_bits), torch.arange(num_bits - 1, -1, -1))

    # Prepare the output tensor
    bitStringTensor = torch.zeros((*uint_tensor.shape[:-1], num_bits))

    # Process each integer in the batch
    for i in range(num_bits):
        div_tensor = torch.div(uint_tensor, powers[i], rounding_mode="trunc")

        uint_tensor = torch.remainder(uint_tensor, powers[i])

        bitStringTensor[..., i] = div_tensor.squeeze()

    return bitStringTensor
