import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
def tensor_to_image(tensor: Float[t.Tensor, "h w c"] | Int[t.Tensor, "h w c"]) -> Image.Image:
    tensor = tensor.clamp(0, 255)
    array = tensor.cpu().numpy().astype(np.uint8)
    array = array[::-1, :, :]
    image = Image.fromarray(array, mode="RGB")
    return image


@jaxtyped(typechecker=typechecker)
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


@jaxtyped(typechecker=typechecker)
def random_unit_vector(shape: tuple[int, ...]) -> Float[t.Tensor, "... 3"]:
    vec = t.randn(*shape)
    vec = F.normalize(vec, dim=-1)
    return vec

@jaxtyped(typechecker=typechecker)
def random_on_hemisphere(normal: Float[t.Tensor, "... 3"]) -> Float[t.Tensor, "... 3"]:
    vec = random_unit_vector(normal.shape)
    dot_product = t.sum(vec * normal, dim=-1, keepdim=True)
    return t.where(dot_product > 0, vec, -vec)
