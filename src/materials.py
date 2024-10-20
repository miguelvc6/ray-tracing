from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hittable import HitRecord
import torch as t
from jaxtyping import Bool, Float, jaxtyped
import torch.nn.functional as F
from typeguard import typechecked as typechecker

from utils import random_unit_vector


@jaxtyped(typechecker=typechecker)
class Material(ABC):
    @jaxtyped(typechecker=typechecker)
    def __init__(self):
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "* 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "* 3"],
        Float[t.Tensor, "* 3 2"],
    ]:
        pass


@jaxtyped(typechecker=typechecker)
class Lambertian(Material):
    def __init__(self, albedo: Float[t.Tensor, "3"]):
        self.albedo = albedo

    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "N 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "* 3"],
        Float[t.Tensor, "* 3 2"],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal
        points = hit_record.point

        # Generate scatter direction
        scatter_direction = normals + random_unit_vector((N, 3))

        # Handle degenerate scatter direction
        zero_mask = scatter_direction.norm(dim=1) < 1e-8
        scatter_direction[zero_mask] = normals[zero_mask]

        # Normalize scatter direction
        scatter_direction = F.normalize(scatter_direction, dim=-1)

        # Create new rays for recursion
        new_origin = points
        new_direction = scatter_direction
        new_rays = t.stack([new_origin, new_direction], dim=-1)

        # Attenuation is the albedo
        attenuation = self.albedo.expand(N, 3)

        scatter_mask = t.ones(N, dtype=t.bool, device=r_in.device)

        return scatter_mask, attenuation, new_rays


@jaxtyped(typechecker=typechecker)
class Metal(Material):
    def __init__(self, albedo: Float[t.Tensor, "3"], fuzz: float = 0.3):
        self.albedo = albedo
        self.fuzz = max(0.0, min(fuzz, 1.0))

    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "N 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "N 3"],
        Float[t.Tensor, "N 3 2"],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal  # Shape: [N, 3]
        points = hit_record.point  # Shape: [N, 3]

        # Incoming ray directions
        in_directions = r_in[:, :, 1]  # Shape: [N, 3]
        in_directions = F.normalize(in_directions, dim=-1)

        # Generate reflected directions
        reflected_direction = in_directions - 2 * t.sum(in_directions * normals, dim=1, keepdim=True) * normals
        reflected_direction = F.normalize(reflected_direction, dim=-1)  # Shape: [N, 3]

        reflected_direction = reflected_direction + self.fuzz * random_unit_vector((N, 3))
        reflected_direction = F.normalize(reflected_direction, dim=-1)

        # Check if reflected ray is above the surface
        dot_product = t.sum(reflected_direction * normals, dim=1)  # Shape: [N]
        scatter_mask = dot_product > 0  # Shape: [N], dtype: bool

        # Create new rays for recursion
        new_origin = points  # Shape: [N, 3]
        new_direction = reflected_direction  # Shape: [N, 3]
        new_rays = t.stack([new_origin, new_direction], dim=-1)  # Shape: [N, 3, 2]

        # Attenuation is the albedo
        attenuation = self.albedo.expand(N, 3)  # Shape: [N, 3]

        return scatter_mask, attenuation, new_rays
