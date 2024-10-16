import numpy as np
import torch as t
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def tensor_to_image(tensor: Float[t.Tensor, "h w c"] | Int[t.Tensor, "h w c"]) -> Image.Image:
    array = tensor.cpu().numpy().astype(np.uint8)
    array = array[::-1, :, :]
    image = Image.fromarray(array, mode="RGB")
    return image


@jaxtyped(typechecker=typechecker)
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


@jaxtyped(typechecker=typechecker)
def random_unit_vector(shape: tuple[int, ...]) -> Float[t.Tensor, "..."]:
    vec = t.randn(*shape)
    return vec / t.norm(vec, dim=-1, keepdim=True)

