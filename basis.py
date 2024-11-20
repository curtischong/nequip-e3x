import math
import jax
import jax.numpy as jnp
import flax.linen as nn


class BesselBasis(nn.Module):
    r_max: float
    num_basis: int = 8
    trainable: bool = True

    def setup(self):
        """Initialize the Bessel basis weights."""
        self.prefactor = 2.0 / self.r_max
        bessel_weights = jnp.linspace(1.0, self.num_basis, self.num_basis) * math.pi

        if self.trainable:
            self.bessel_weights = self.param("bessel_weights", lambda _: bessel_weights)
        else:
            self.bessel_weights = bessel_weights

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : jnp.ndarray
            Input array

        Returns
        -------
        jnp.ndarray
            Bessel basis evaluation
        """
        # Add dimension for broadcasting
        x_expanded = jnp.expand_dims(x, axis=-1)

        # Calculate the basis functions
        numerator = jnp.sin(self.bessel_weights * x_expanded / self.r_max)
        return self.prefactor * (numerator / x_expanded)
