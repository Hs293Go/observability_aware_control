from typing import Any, Callable

import jax
from jax.typing import ArrayLike

DynamicsFunction = Callable[[ArrayLike, ArrayLike], jax.Array]
ObservationFunction = Callable[[ArrayLike, ArrayLike, Any], jax.Array]
OutputFunction = Callable[[ArrayLike], jax.Array]
