import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker

from config import device


@jaxtyped(typechecker=typechecker)
def tensor_to_image(tensor: Float[t.Tensor, "h w c"] | Int[t.Tensor, "h w c"]) -> Image.Image:
    tensor = tensor.sqrt()  # gamma correction
    tensor = tensor.multiply(255).clamp(0, 255)
    array = tensor.cpu().numpy().astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    return image


@jaxtyped(typechecker=typechecker)
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


@jaxtyped(typechecker=typechecker)
def random_unit_vector(shape: tuple[int, ...]) -> Float[t.Tensor, "... 3"]:
    vec = t.randn(*shape, device=device)
    vec = F.normalize(vec, dim=-1)
    return vec


@jaxtyped(typechecker=typechecker)
def random_on_hemisphere(normal: Float[t.Tensor, "... 3"]) -> Float[t.Tensor, "... 3"]:
    vec = random_unit_vector(normal.shape)
    dot_product = t.sum(vec * normal, dim=-1, keepdim=True)
    return t.where(dot_product > 0, vec, -vec)


@jaxtyped(typechecker=typechecker)
def background_color_gradient(sample: int, h: int, w: int) -> Float[t.Tensor, "sample h w 3"]:
    white: Float[t.Tensor, "3"] = t.tensor([1.0, 1.0, 1.0], device=device)
    light_blue: Float[t.Tensor, "3"] = t.tensor([0.5, 0.7, 1.0], device=device)
    a: Float[t.Tensor, "h 1"] = t.linspace(0, 1, h, device=device).unsqueeze(1)
    background_colors_single: Float[t.Tensor, "h 3"] = a * light_blue + (1.0 - a) * white
    background_colors: Float[t.Tensor, "sample h w 3"] = (
        background_colors_single.unsqueeze(0).unsqueeze(2).expand(sample, h, w, 3) * 255
    )
    return background_colors


@jaxtyped(typechecker=typechecker)
def random_in_unit_disk(shape: tuple[int, ...]) -> Float[t.Tensor, "... 2"]:
    r: Float[t.Tensor, "..."] = t.sqrt(t.rand(*shape, device=device))
    theta: Float[t.Tensor, "..."] = t.rand(*shape, device=device) * 2 * np.pi
    x: Float[t.Tensor, "..."] = r * t.cos(theta)
    y: Float[t.Tensor, "..."] = r * t.sin(theta)
    return t.stack([x, y], dim=-1)
