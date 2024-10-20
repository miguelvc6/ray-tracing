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
        pixel_rays: Float[t.Tensor, "sample h w 3 2"],
        world: Hittable,
        depth: int = 50,
    ) -> Float[t.Tensor, "sample h w 3"]:
        if depth <= 0:
            # Return black when maximum depth is reached
            N = pixel_rays.shape[0] * pixel_rays.shape[1] * pixel_rays.shape[2]
            return t.zeros((N, 3), dtype=pixel_rays.dtype, device=pixel_rays.device).view(pixel_rays.shape[:-2] + (3,))

        # Flatten pixel_rays
        original_shape = pixel_rays.shape[:-2]
        N = pixel_rays.numel() // (3 * 2)
        pixel_rays_flat: Float[t.Tensor, "N 3 2"] = pixel_rays.view(N, 3, 2)

        # Perform hit test
        hit_record: HitRecord = world.hit(pixel_rays_flat, 0.001, float("inf"))

        # Initialize colors
        colors: Float[t.Tensor, "N 3"] = t.zeros((N, 3), dtype=pixel_rays.dtype, device=pixel_rays.device)

        # Handle rays that did not hit anything
        no_hit_mask: Bool[t.Tensor, "N"] = ~hit_record.hit
        if no_hit_mask.any():
            # Compute background color based on ray direction
            ray_dirs: Float[t.Tensor, "N 3"] = F.normalize(pixel_rays_flat[:, :, 1], dim=-1)
            t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
            background_colors_flat = (1.0 - t_param).unsqueeze(-1) * t.tensor([1.0, 1.0, 1.0])
            background_colors_flat += t_param.unsqueeze(-1) * t.tensor([0.5, 0.7, 1.0])
            colors[no_hit_mask] = background_colors_flat[no_hit_mask]

        # Handle rays that hit an object
        hit_mask: Bool[t.Tensor, "N"] = hit_record.hit
        M = hit_mask.sum()  # Number of hits
        if hit_mask.any():
            normals: Float[t.Tensor, "M 3"] = hit_record.normal[hit_mask]
            points: Float[t.Tensor, "M 3"] = hit_record.point[hit_mask]

            # Generate scatter direction with lambertian reflection
            scatter_direction: Float[t.Tensor, "M 3"] = normals + random_unit_vector(normals.shape)

            # Handle degenerate scatter direction
            zero_mask: Bool[t.Tensor, "M"] = scatter_direction.norm(dim=1) < 1e-8
            scatter_direction[zero_mask] = normals[zero_mask]

            # Normalize scatter direction
            scatter_direction = F.normalize(scatter_direction, dim=-1)

            # Create new rays for recursion
            new_origin: Float[t.Tensor, "M 3"] = points
            new_direction: Float[t.Tensor, "M 3"] = scatter_direction
            new_rays: Float[t.Tensor, "M 3 2"] = t.stack([new_origin, new_direction], dim=-1)

            # Recursive call
            new_colors: Float[t.Tensor, "M 3"] = self.ray_color(
                new_rays.view(-1, 1, 1, 3, 2),
                world,
                depth - 1,
            ).view(-1, 3)

            # Attenuate color for diffuse reflection
            colors[hit_mask] = 0.5 * new_colors

        # Reshape colors back to [sample, h, w, 3]
        colors = colors.view(*original_shape, 3)
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
        colors: Float[t.Tensor, "sample h w 3"] = self.ray_color(pixel_rays, world, depth=50)

        # Average over antialiasing samples
        img: Float[t.Tensor, "h w 3"] = colors.mean(dim=0)

        # Convert to image
        return tensor_to_image(img)
