import torch
import torch.nn.functional as F
from torch import nn

from .helpers import default, log, extract


class GaussianDiffusion(nn.Module):
    """
    `Diffusion Model <https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/>`_.
    """

    def __init__(
            self,
            *,
            timesteps: int
    ):
        """
        :param timesteps: Number of timesteps in the Diffusion Process.
        """
        super().__init__()

        # Timesteps < 20 => scale > 50 => beta_end > 1 => alphas[-1] < 0 => sqrt_alphas_cumprod[-1] is NaN
        assert not timesteps < 20,  f'timsteps must be at least 20'
        self.num_timesteps = timesteps

        # Create variance schedule.
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        # Diffusion model constants/buffers. See https://arxiv.org/pdf/2006.11239.pdf
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # register buffer helper function
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # Register variance schedule related buffers
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Buffer for diffusion calculations q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # Posterior variance:
        #   https://github.com/AssemblyAI-Examples/build-your-own-imagen/blob/main/images/posterior_variance.png
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Clipped because posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps=1e-20))

        # Buffers for calculating the q_posterior mean $\~{\mu}$. See
        #   https://github.com/oconnoob/minimal_imagen/blob/minimal/images/posterior_mean_coeffs.png
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _get_times(self, batch_size: int, noise_level: float, *, device: torch.device) -> torch.tensor:
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device=device, dtype=torch.long)

    def _sample_random_times(self, batch_size: int, *, device: torch.device) -> torch.tensor:
        """
        Randomly sample `batch_size` timestep values uniformly from [0, 1, ..., `self.num_timesteps`]

        :param batch_size: Number of images in the batch.
        :param device: Device on which to place the return tensor.
        :return: Tensor of integers (`dtype=torch.long`) of shape `(batch_size,)`
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def _get_sampling_timesteps(self, batch: int, *, device: torch.device) -> list[torch.tensor]:
        time_transitions = []

        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch,), i, device=device, dtype=torch.long)))

        return time_transitions

    def q_posterior(self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor) -> tuple[torch.tensor,
                                                                                              torch.tensor,
                                                                                              torch.tensor]:
        """
        Calculates q_posterior parameters given a starting image :code:`x_start` (x_0) and a noised image :code:`x_t`.

        .. figure:: _static/q_posterior.png
            :alt: q posterior formula

        Where the mean is

        .. image:: _static/posterior_mean.png

        And the variance prefactor is

        .. image:: _static/posterior_variance.png
            :width: 300px

        :param x_start: Original input images x_0. Shape (b, c, h, w)
        :param x_t: Images at current time x_t. Shape (b, c, h, w)
        :param t: Current time. Shape (b,)
        :return: Tuple of

            - **posterior mean**, shape (b, c, s, s),

            - **posterior variance**, shape (b, 1, 1, 1),

            - **clipped log of the posterior variance**, shape (b, 1, 1, 1)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Extract the value corresponding to the current time from the buffers, and then reshape to (b, 1, 1, 1)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start: torch.tensor, t: torch.tensor, noise: torch.tensor = None) -> torch.tensor:
        """
        Sample from q at a given timestep:

            .. image:: _static/q_sample.png


        :param x_start: Original input images. Shape (b, c, h, w).
        :param t: Timestep value for each image in the batch. Shape (b,).
        :param noise: Optionally supply noise to use. Defaults to Gaussian. Shape (b, c, s, s).
        :return: Noised image. Shape (b, c, h, w).

        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return noised

    def predict_start_from_noise(self, x_t: torch.tensor, t: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Given a noised image and its noise component, calculated the unnoised image :code:`x_0`.

        :param x_t: Noised images. Shape (b, c, s, s).
        :param t: Timestep for each image. Shape (b,).
        :param noise: Noise component for each image. Shape (b, c, s, s).
        :return: Un-noised images. Shape (b, c, s, s).

        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
