"""Flax modules for force estimation."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class ForceEstimator(nn.Module):
    """Two-layer MLP that predicts 3D force vectors from observations."""

    hidden_size: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size)(x)
        x = nn.elu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(self.hidden_size)(x)
        x = nn.elu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(3)(x)
        return x



