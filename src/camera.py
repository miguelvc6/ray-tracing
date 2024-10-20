import math

import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker

from hittable import HitRecord, Hittable
from utils import degrees_to_radians, random_in_unit_disk, tensor_to_image

device = t.device("cuda" if t.cuda.is_available() else "cpu")

@jaxtyped(typechecker=typechecker)
class Camera:
    def __init__(
        self,
        look_from: Float[t.Tensor, "3"] = t.tensor([0.0, 0.0, 0.0], device=device),
        look_at: Float[t.Tensor, "3"] = t.tensor([0.0, 0.0, -1.0], device=device),
        vup: Float[t.Tensor, "3"] = t.tensor([0.0, 1.0, 0.0], device=device),
        aspect_ratio: float = 16.0 / 9.0,
        image_width: int = 400,
        samples_per_pixel: int = 10,
        max_depth: int = 50,
        vfov: float = 90.0,  # vertical field of view angle
        defocus_angle: float = 0,
        focus_dist: float = 10.0,
    ):
        self.look_from: Float[t.Tensor, "3"] = look_from
        self.look_at: Float[t.Tensor, "3"] = look_at
        self.vup: Float[t.Tensor, "3"] = vup

        self.samples_per_pixel: int = samples_per_pixel
        self.max_depth: int = max_depth

        self.defocus_angle: float = defocus_angle
        self.focus_dist: float = focus_dist

        self.aspect_ratio: float = aspect_ratio
        self.image_width: int = image_width
        self.image_height: int = max(1, int(image_width / aspect_ratio))
        h, w = self.image_height, self.image_width

        # Compute viewport dimensions
        theta: float = degrees_to_radians(vfov)
        h_viewport: float = math.tan(theta / 2)
        self.viewport_height: float = 2.0 * h_viewport * focus_dist
        self.viewport_width: float = self.viewport_height * self.aspect_ratio

        # Calculate camera basis vectors
        self.w: Float[t.Tensor, "3"] = F.normalize(self.look_from - self.look_at, dim=-1)
        self.u: Float[t.Tensor, "3"] = F.normalize(t.cross(self.vup, self.w, dim=-1), dim=-1)
        self.v: Float[t.Tensor, "3"] = t.cross(self.w, self.u, dim=-1)

        # Calculate viewport dimensions and vectors
        viewport_u: Float[t.Tensor, "3"] = self.viewport_width * self.u
        viewport_v: Float[t.Tensor, "3"] = self.viewport_height * self.v

        # Calculate the lower-left corner of the viewport
        self.viewport_lower_left: Float[t.Tensor, "3"] = (
            self.look_from - viewport_u / 2 - viewport_v / 2 - self.w * focus_dist
        )

        # Calculate the camera defocus disk basis vectors
        defocus_radius: float = focus_dist * math.tan(degrees_to_radians(defocus_angle / 2))
        self.defocus_disk_u: Float[t.Tensor, "3"] = self.u * defocus_radius
        self.defocus_disk_v: Float[t.Tensor, "3"] = self.v * defocus_radius

        # Store pixel delta vectors
        self.pixel_delta_u: Float[t.Tensor, "3"] = viewport_u / (w - 1)
        self.pixel_delta_v: Float[t.Tensor, "3"] = viewport_v / (h - 1)

        self.viewport_lower_left = self.viewport_lower_left.to(device)
        self.pixel_delta_u = self.pixel_delta_u.to(device)
        self.pixel_delta_v = self.pixel_delta_v.to(device)


    @jaxtyped(typechecker=typechecker)
    def defocus_disk_sample(self, sample: int, h: int, w: int) -> Float[t.Tensor, "sample h w 3"]:
        p: Float[t.Tensor, "sample h w 2"] = random_in_unit_disk((sample, h, w))
        offset = p[..., 0].unsqueeze(-1) * self.defocus_disk_u.view(1, 1, 1, 3) + p[..., 1].unsqueeze(
            -1
        ) * self.defocus_disk_v.view(1, 1, 1, 3)
        return self.look_from.view(1, 1, 1, 3) + offset

    @jaxtyped(typechecker=typechecker)
    def ray_color(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        world: Hittable,
        depth: int = 50,
    ) -> Float[t.Tensor, "N 3"]:
        if depth <= 0:
            # Return black when maximum depth is reached
            return t.zeros((pixel_rays.shape[0], 3), device=device)

        N = pixel_rays.shape[0]
        # Perform hit test
        hit_record: HitRecord = world.hit(pixel_rays, 0.001, float("inf"))

        # Initialize colors
        colors: Float[t.Tensor, "N 3"] = t.zeros((pixel_rays.shape[0], 3), device=device)

        # Handle rays that did not hit anything
        no_hit_mask: Bool[t.Tensor, "N"] = ~hit_record.hit
        if no_hit_mask.any():
            # Compute background color based on ray direction
            ray_dirs: Float[t.Tensor, "N 3"] = F.normalize(pixel_rays[:, :, 1], dim=-1)
            t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
            background_colors_flat = (1.0 - t_param).unsqueeze(-1) * t.tensor([1.0, 1.0, 1.0], device=device)
            background_colors_flat += t_param.unsqueeze(-1) * t.tensor([0.5, 0.7, 1.0], device=device)
            colors[no_hit_mask] = background_colors_flat[no_hit_mask]

        # Handle rays that hit an object
        hit_mask: Bool[t.Tensor, "N"] = hit_record.hit
        if hit_mask.any():
            hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(-1)
            materials = [hit_record.material[idx] for idx in hit_indices.tolist()]

            # Group indices by material
            material_to_indices = {}
            for idx, material in zip(hit_indices.tolist(), materials):
                if material not in material_to_indices:
                    material_to_indices[material] = []
                material_to_indices[material].append(idx)

            for material, indices in material_to_indices.items():
                indices = t.tensor(indices, dtype=t.long, device=device)
                ray_in = pixel_rays[indices]
                sub_hit_record = HitRecord(
                    hit=hit_record.hit[indices],
                    point=hit_record.point[indices],
                    normal=hit_record.normal[indices],
                    t=hit_record.t[indices],
                    front_face=hit_record.front_face[indices],
                    material=[material] * len(indices),
                )

                scatter_mask, attenuation, scattered_rays = material.scatter(ray_in, sub_hit_record)

                # Separate indices for rays that scatter and rays that do not
                scatter_indices = scatter_mask.nonzero(as_tuple=False).squeeze(-1)
                no_scatter_indices = (~scatter_mask).nonzero(as_tuple=False).squeeze(-1)

                # Handle scattered rays
                if scatter_indices.numel() > 0:
                    recursive_colors = self.ray_color(scattered_rays[scatter_indices], world, depth - 1)
                    colors[indices[scatter_indices]] = attenuation[scatter_indices] * recursive_colors

                # Handle rays that do not scatter
                if no_scatter_indices.numel() > 0:
                    colors[indices[no_scatter_indices]] = t.zeros(
                        (no_scatter_indices.numel(), 3), device=device
                    )

        return colors

    @jaxtyped(typechecker=typechecker)
    def render(self, world: Hittable) -> Image.Image:
        sample: int = self.samples_per_pixel
        h, w = self.image_height, self.image_width

        # Prepare grid indices for pixels
        j_indices = t.arange(h, device=device).view(1, h, 1, 1)
        i_indices = t.arange(w, device=device).view(1, 1, w, 1)

        # Generate random offsets for antialiasing
        noise_u: Float[t.Tensor, "sample h w 1"] = t.rand((sample, h, w, 1), device=device)
        noise_v: Float[t.Tensor, "sample h w 1"] = t.rand((sample, h, w, 1), device=device)

        # Compute pixel positions on the viewport
        sampled_pixels: Float[t.Tensor, "sample h w 3"] = (
            self.viewport_lower_left.view(1, 1, 1, 3)
            + (i_indices + noise_u) * self.pixel_delta_u.view(1, 1, 1, 3)
            + (j_indices + noise_v) * self.pixel_delta_v.view(1, 1, 1, 3)
        )

        # Compute direction vectors
        directions: Float[t.Tensor, "sample h w 3"] = F.normalize(
            sampled_pixels - self.look_from.to(device).view(1, 1, 1, 3), dim=-1
        )

        # Build rays
        origin: Float[t.Tensor, "sample h w 3"] = self.look_from.to(device).view(1, 1, 1, 3).expand(sample, h, w, 3)
        if self.defocus_angle <= 0:
            ray_origin = origin
        else:
            ray_origin = self.defocus_disk_sample(sample, h, w)

        pixel_rays: Float[t.Tensor, "sample h w 3 2"] = t.stack([ray_origin, directions], dim=-1)

        # Flatten rays for processing
        N = sample * h * w
        pixel_rays_flat: Float[t.Tensor, "N 3 2"] = pixel_rays.view(N, 3, 2)

        # Call ray_color
        colors_flat: Float[t.Tensor, "N 3"] = self.ray_color(pixel_rays_flat, world, depth=self.max_depth)
        colors: Float[t.Tensor, "sample h w 3"] = colors_flat.view(sample, h, w, 3)

        # Average over antialiasing samples
        img: Float[t.Tensor, "h w 3"] = colors.mean(dim=0)

        # Convert to image
        return tensor_to_image(img)
