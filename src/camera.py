import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker

from hittable import HitRecord, Hittable
from utils import tensor_to_image, random_unit_vector


@jaxtyped(typechecker=typechecker)
class Camera:
    def __init__(
        self,
        origin: Float[t.Tensor, "3"] = t.Tensor([0.0, 0.0, 0.0]),
        focal_length: float = 1.0,
        aspect_ratio: float = 16.0 / 9.0,
        image_width: int = 400,
        viewport_height: float = 2.0,
        samples_per_pixel: int = 10,
    ):
        self.origin: Float[t.Tensor, "3"] = origin
        self.focal_length: float = focal_length
        self.samples_per_pixel: int = samples_per_pixel

        self.aspect_ratio: float = aspect_ratio
        self.image_width: int = image_width
        self.image_height: int = max(1, int(image_width / aspect_ratio))
        h, w = self.image_height, self.image_width

        self.viewport_height: float = viewport_height
        self.viewport_width: float = viewport_height * (w / h)

        # x-axis
        left_border: float = -self.viewport_width / 2
        self.width_pixel_delta: float = self.viewport_width / w
        pixels_x: Float[t.Tensor, "w"] = t.linspace(
            left_border + self.width_pixel_delta / 2, -left_border - self.width_pixel_delta / 2, w
        )

        # y-axis
        top_border: float = viewport_height / 2
        self.height_pixel_delta: float = viewport_height / h
        pixels_y: Float[t.Tensor, "h"] = t.linspace(
            -top_border + self.height_pixel_delta / 2, top_border - self.height_pixel_delta / 2, h
        )

        # z-axis
        pixels_z: Float[t.Tensor, "1"] = t.Tensor([-focal_length])

        # Generate coordinate grids
        grid_x, grid_y, grid_z = t.meshgrid(pixels_x, pixels_y, pixels_z, indexing="ij")
        viewport_pixels: Float[t.Tensor, "h w 3"] = t.stack([grid_x, grid_y, grid_z], dim=-1)
        self.viewport_pixels = viewport_pixels.permute(1, 0, 2, 3).reshape(h, w, 3)

    @jaxtyped(typechecker=typechecker)
    def ray_color(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        world: Hittable,
        depth: int = 50,
    ) -> Float[t.Tensor, "N 3"]:
        if depth <= 0:
            # Return black when maximum depth is reached
            return t.zeros((pixel_rays.shape[0], 3), device=pixel_rays.device)

        N = pixel_rays.shape[0]
        # Perform hit test
        hit_record: HitRecord = world.hit(pixel_rays, 0.001, float("inf"))

        # Initialize colors
        colors: Float[t.Tensor, "N 3"] = t.zeros((pixel_rays.shape[0], 3), device=pixel_rays.device)

        # Handle rays that did not hit anything
        no_hit_mask: Bool[t.Tensor, "N"] = ~hit_record.hit
        if no_hit_mask.any():
            # Compute background color based on ray direction
            ray_dirs: Float[t.Tensor, "N 3"] = F.normalize(pixel_rays[:, :, 1], dim=-1)
            t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
            background_colors_flat = (1.0 - t_param).unsqueeze(-1) * t.tensor([1.0, 1.0, 1.0])
            background_colors_flat += t_param.unsqueeze(-1) * t.tensor([0.5, 0.7, 1.0])
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
                indices = t.tensor(indices, dtype=t.long, device=pixel_rays.device)
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

                if scatter_mask.any():
                    scatter_indices = scatter_mask.nonzero(as_tuple=False).squeeze(-1)
                    recursive_colors = self.ray_color(scattered_rays[scatter_indices], world, depth - 1)
                    colors[indices[scatter_indices]] = attenuation[scatter_indices] * recursive_colors
                else:
                    colors[indices] = t.zeros((len(indices), 3), device=pixel_rays.device)

        return colors

    @jaxtyped(typechecker=typechecker)
    def render(self, world: Hittable) -> Image.Image:
        sample: int = self.samples_per_pixel
        h, w = self.image_height, self.image_width

        # Antialiasing: Generate multiple samples per pixel
        noise: Float[t.Tensor, "sample h w 2"] = t.rand((sample, h, w, 2)) - 0.5
        scale: Float[t.Tensor, "2"] = t.tensor([self.width_pixel_delta, self.height_pixel_delta])
        noise = noise * scale.view(1, 1, 1, 2)

        # Adjust pixel positions
        sampled_pixels_xy: Float[t.Tensor, "sample h w 2"] = self.viewport_pixels[:, :, :2].unsqueeze(0) + noise
        sampled_pixels_z: Float[t.Tensor, "sample h w 1"] = (
            self.viewport_pixels[:, :, 2].unsqueeze(0).unsqueeze(-1).expand(sample, h, w, 1)
        )
        sampled_pixels: Float[t.Tensor, "sample h w 3"] = t.cat([sampled_pixels_xy, sampled_pixels_z], dim=-1)

        # Compute direction vectors
        directions: Float[t.Tensor, "sample h w 3"] = F.normalize(sampled_pixels - self.origin, dim=-1)

        # Normalize direction vectors
        directions = F.normalize(directions, dim=-1)

        # Build rays
        origin: Float[t.Tensor, "sample h w 3"] = self.origin.view(1, 1, 1, 3).expand(sample, h, w, 3)
        pixel_rays: Float[t.Tensor, "sample h w 3 2"] = t.stack([origin, directions], dim=-1)

        # Call ray_color
        N = pixel_rays.numel() // (3 * 2)
        pixel_rays_flat: Float[t.Tensor, "N 3 2"] = pixel_rays.view(N, 3, 2)
        colors_flat: Float[t.Tensor, "N 3"] = self.ray_color(pixel_rays_flat, world, depth=50)
        colors: Float[t.Tensor, "sample h w 3"] = colors_flat.view(sample, h, w, 3)

        # Average over antialiasing samples
        img: Float[t.Tensor, "h w 3"] = colors.mean(dim=0)

        # Convert to image
        return tensor_to_image(img)
