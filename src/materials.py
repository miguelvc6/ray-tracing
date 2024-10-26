from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hittable import HitRecord
from enum import IntEnum

import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, jaxtyped
from typeguard import typechecked as typechecker

from config import device
from utils import random_unit_vector


class MaterialType(IntEnum):
    Lambertian = 0
    Metal = 1
    Dielectric = 2


@jaxtyped(typechecker=typechecker)
def reflect(v: Float[t.Tensor, "N 3"], n: Float[t.Tensor, "N 3"]) -> Float[t.Tensor, "N 3"]:
    # Reflects vector v around normal n
    return v - 2 * (v * n).sum(dim=1, keepdim=True) * n


@jaxtyped(typechecker=typechecker)
def refract(
    uv: Float[t.Tensor, "N 3"], n: Float[t.Tensor, "N 3"], etai_over_etat: Float[t.Tensor, "N 1"]
) -> Float[t.Tensor, "N 3"]:
    one = t.tensor(1.0, device=uv.device)
    cos_theta = t.minimum((-uv * n).sum(dim=1, keepdim=True), one)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -t.sqrt(t.abs(one - (r_out_perp**2).sum(dim=1, keepdim=True))) * n
    return r_out_perp + r_out_parallel


@jaxtyped(typechecker=typechecker)
def reflectance(cosine: Float[t.Tensor, "N 1"], ref_idx: Float[t.Tensor, "N 1"]) -> Float[t.Tensor, "N 1"]:
    one = t.tensor(1.0, device=ref_idx.device)
    r0 = ((one - ref_idx) / (one + ref_idx)) ** 2
    return r0 + (one - r0) * (one - cosine) ** 5


@jaxtyped(typechecker=typechecker)
class Material(ABC):
    @jaxtyped(typechecker=typechecker)
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def scatter_material(
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
    @jaxtyped(typechecker=typechecker)
    def __init__(self, albedo: Float[t.Tensor, "3"]):
        self.albedo = albedo.to(device)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def scatter_material(
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
        scatter_direction = normals + random_unit_vector((N, 3)).to(device)

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
        attenuation = hit_record.albedo
        scatter_mask = t.ones(N, dtype=t.bool, device=device)

        return scatter_mask, attenuation, new_rays


@jaxtyped(typechecker=typechecker)
class Metal(Material):
    @jaxtyped(typechecker=typechecker)
    def __init__(self, albedo: Float[t.Tensor, "3"], fuzz: float = 0.3):
        self.albedo = albedo.to(device)
        self.fuzz = max(0.0, min(fuzz, 1.0))

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def scatter_material(
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
        fuzz = hit_record.fuzz.unsqueeze(1)  # Shape: [N, 1]

        reflected_direction = reflect(in_directions, normals)
        reflected_direction = reflected_direction + fuzz * random_unit_vector((N, 3)).to(device)
        reflected_direction = F.normalize(reflected_direction, dim=-1)

        # Check if reflected ray is above the surface
        dot_product = t.sum(reflected_direction * normals, dim=1)  # Shape: [N]
        scatter_mask = dot_product > 0  # Shape: [N], dtype: bool

        # Create new rays for recursion
        new_origin = points  # Shape: [N, 3]
        new_direction = reflected_direction  # Shape: [N, 3]
        new_rays = t.stack([new_origin, new_direction], dim=-1)  # Shape: [N, 3, 2]

        # Attenuation is the albedo
        attenuation = hit_record.albedo

        return scatter_mask, attenuation, new_rays


@jaxtyped(typechecker=typechecker)
class Dielectric(Material):
    def __init__(self, refraction_index: float):
        self.refraction_index = refraction_index

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def scatter_material(
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
        front_face = hit_record.front_face  # Shape: [N], dtype: bool
        unit_direction = F.normalize(r_in[:, :, 1], dim=1)  # Shape: [N, 3]

        # Attenuation is always (1, 1, 1) for dielectric materials
        attenuation = t.ones(N, 3, device=device)  # Shape: [N, 3]

        one = t.tensor(1.0, device=device)
        refractive_indices = hit_record.refractive_index.unsqueeze(1)  # Shape: [N, 1]
        refraction_ratio = t.where(
            front_face.unsqueeze(1),
            one / refractive_indices,
            refractive_indices,
        )

        cos_theta = t.minimum((-unit_direction * normals).sum(dim=1, keepdim=True), one)
        sin_theta = t.sqrt(one - cos_theta**2)

        cannot_refract = (refraction_ratio * sin_theta) > one

        # Generate random numbers to decide between reflection and refraction
        reflect_prob = reflectance(cos_theta, refraction_ratio)
        random_numbers = t.rand(N, 1, device=device)
        should_reflect = cannot_refract | (reflect_prob > random_numbers)

        # Compute reflected and refracted directions
        reflected_direction = reflect(unit_direction, normals)
        refracted_direction = refract(unit_direction, normals, refraction_ratio)
        direction = t.where(should_reflect.expand(-1, 3), reflected_direction, refracted_direction)
        new_rays = t.stack([points, direction], dim=-1)

        # Scatter mask is always True for dielectric materials
        scatter_mask = t.ones(N, dtype=t.bool, device=device)

        return scatter_mask, attenuation, new_rays
