#!/usr/bin/env python3
"""Convert old policy checkpoints to the new positional encoding structure.

The old checkpoint uses:
    attn_*_net.position_embedding.penc.inv_freq

The new checkpoint uses:
    attn_*_net.pos_encoding

This script converts old checkpoints to the new format by:
1. Removing the old position_embedding.penc.inv_freq keys
2. Computing and adding the new pos_encoding tensors

Usage:
    ./isaaclab.sh -p scripts/convert_checkpoint.py --input <old_checkpoint.pt> --output <new_checkpoint.pt>

    # Or convert in-place:
    ./isaaclab.sh -p scripts/convert_checkpoint.py --input <checkpoint.pt> --inplace
"""

import argparse
import math
from pathlib import Path

import torch


def _compute_positional_encoding_3d(
    channels: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute 3D positional encoding for given spatial dimensions.

    This code is adapted from: https://github.com/tatp22/multidim-positional-encoding

    Args:
        channels: Number of channels for the encoding.
        D, H, W: Spatial dimensions (depth, height, width).
        device: Device to create tensor on.
        dtype: Data type for the encoding tensor.

    Returns:
        Positional encoding tensor of shape (1, channels, D, H, W).
    """
    org_channels = channels
    channels = int(math.ceil(channels / 6) * 2)
    if channels % 2:
        channels += 1
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device).float() / channels))

    def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    pos_x = torch.arange(D, device=device, dtype=inv_freq.dtype)
    pos_y = torch.arange(H, device=device, dtype=inv_freq.dtype)
    pos_z = torch.arange(W, device=device, dtype=inv_freq.dtype)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
    emb_y = get_emb(sin_inp_y).unsqueeze(1)
    emb_z = get_emb(sin_inp_z)
    emb = torch.zeros((D, H, W, channels * 3), device=device, dtype=dtype)
    emb[:, :, :, :channels] = emb_x
    emb[:, :, :, channels : 2 * channels] = emb_y
    emb[:, :, :, 2 * channels :] = emb_z

    # Convert from (D, H, W, ch) to (1, ch, D, H, W) format (channel-first with batch dim)
    enc = emb[None, :, :, :, :org_channels]  # (1, D, H, W, org_channels)
    enc = enc.permute(0, 4, 1, 2, 3)  # (1, org_channels, D, H, W)
    return enc


def convert_checkpoint(
    input_path: str,
    output_path: str,
    image_input_dims: tuple = (64, 5, 8),
    height_input_dims: tuple = (64, 7, 7),
    num_cameras: int = 1,
    force: bool = False,
) -> None:
    """Convert old checkpoint to new format.

    Args:
        input_path: Path to the old checkpoint.
        output_path: Path to save the converted checkpoint.
        image_input_dims: Tuple of (C, H, W) for image input.
        height_input_dims: Tuple of (C, H, W) for height map input.
        num_cameras: Number of cameras (1 or 2).
        force: Force re-compute pos_encoding even if already in new format.
    """
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]

    # Find old position_embedding keys to remove
    old_keys = [k for k in state_dict.keys() if "position_embedding.penc.inv_freq" in k]

    # Check if new keys exist
    new_keys = [k for k in state_dict.keys() if ".pos_encoding" in k]

    if not old_keys and not force:
        print("No old position_embedding keys found. Checkpoint may already be in new format.")
        if new_keys:
            print(f"Found new pos_encoding keys: {new_keys}")
            for k in new_keys:
                print(f"  {k}: {state_dict[k].shape}")
            print("Checkpoint is already in new format. Use --force to re-compute pos_encoding.")
            return
        else:
            print("Warning: Neither old nor new positional encoding keys found!")
            return

    if old_keys:
        print(f"Found {len(old_keys)} old position_embedding keys to convert:")
        for k in old_keys:
            print(f"  - {k}")
        # Remove old keys
        for k in old_keys:
            del state_dict[k]

    if new_keys and force:
        print(f"Force mode: Replacing {len(new_keys)} existing pos_encoding keys:")
        for k in new_keys:
            print(f"  - {k}: {state_dict[k].shape}")
        # Remove existing pos_encoding keys to replace them
        for k in new_keys:
            del state_dict[k]

    # Compute and add new positional encodings
    # For attn_image_net and attn_critic_image_net: spatial_dims = (num_cameras, H, W)
    # For attn_height_net: spatial_dims = (1, H, W)

    image_channels = image_input_dims[0]
    height_channels = height_input_dims[0]

    image_spatial_dims = (num_cameras, image_input_dims[1], image_input_dims[2])
    height_spatial_dims = (1, height_input_dims[1], height_input_dims[2])

    print(f"\nComputing new positional encodings:")
    print(f"  - image_input_dims: {image_input_dims}, num_cameras: {num_cameras}")
    print(f"  - height_input_dims: {height_input_dims}")
    print(f"  - image spatial_dims (D, H, W): {image_spatial_dims}")
    print(f"  - height spatial_dims (D, H, W): {height_spatial_dims}")

    # Compute positional encodings
    image_pos_enc = _compute_positional_encoding_3d(
        image_channels, *image_spatial_dims, torch.device("cpu"), torch.float32
    )
    height_pos_enc = _compute_positional_encoding_3d(
        height_channels, *height_spatial_dims, torch.device("cpu"), torch.float32
    )

    print(f"\nNew positional encoding shapes:")
    print(f"  - attn_image_net.pos_encoding: {image_pos_enc.shape}")
    print(f"  - attn_height_net.pos_encoding: {height_pos_enc.shape}")
    print(f"  - attn_critic_image_net.pos_encoding: {image_pos_enc.shape}")

    # Add new keys
    state_dict["attn_image_net.pos_encoding"] = image_pos_enc
    state_dict["attn_height_net.pos_encoding"] = height_pos_enc
    state_dict["attn_critic_image_net.pos_encoding"] = image_pos_enc.clone()

    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert old policy checkpoints to new positional encoding structure."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the old checkpoint file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the converted checkpoint. If not specified, uses --inplace."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Convert the checkpoint in-place (overwrite the input file)."
    )
    parser.add_argument(
        "--image-dims",
        type=str,
        default="64,5,8",
        help="Image input dimensions as C,H,W (default: 64,5,8)."
    )
    parser.add_argument(
        "--height-dims",
        type=str,
        default="64,7,7",
        help="Height input dimensions as C,H,W (default: 64,7,7)."
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=1,
        help="Number of cameras (default: 1)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-compute pos_encoding even if already in new format."
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.inplace:
        output_path = args.input
    else:
        # Default: add _converted suffix
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_converted{input_path.suffix}")

    # Parse dimensions
    image_dims = tuple(int(x) for x in args.image_dims.split(","))
    height_dims = tuple(int(x) for x in args.height_dims.split(","))

    if len(image_dims) != 3 or len(height_dims) != 3:
        raise ValueError("Dimensions must be specified as C,H,W (3 values)")

    convert_checkpoint(
        input_path=args.input,
        output_path=output_path,
        image_input_dims=image_dims,
        height_input_dims=height_dims,
        num_cameras=args.num_cameras,
        force=args.force,
    )


if __name__ == "__main__":
    main()
