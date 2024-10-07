"""
Copyright © 2024 H S Helson Go and Ching Lok Chong

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Literal, Mapping, Optional, Union

import numpy as np
from matplotlib import animation, axes, figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import ArrayLike


class AnimatedRobotTrajectory:
    """A wrapper over FuncAnimation that manages animation of 3D robot trajectories"""

    def __init__(
        self,
        fig: figure.Figure,
        ax: Union[axes.Axes, Axes3D],
        dims: Literal[2, 3] = 2,
        animation_kws: Optional[Mapping] = None,
    ) -> None:
        """Initializes the robot trajectory animator

        Parameters
        ----------
        fig : figure.Figure
            The figure that will be animated
        ax : Union[axes.Axes, Axes3D]
            The axes that will be drawn upon, may be 3D axes
        dims : Literal[2, 3], optional
            Number of dimensions of the plot, by default 2
        animation_kws : Optional[Mapping], optional
            Keyword arguments to be passed into the FuncAnimation object, by
            default None

        Raises
        ------
        ValueError
            If `dims` is neither 2 nor 3

        """
        self.annotation = ""
        self.x = np.array([])
        self.y = np.array([])
        if dims == 3:
            self.z = np.array([])
        elif dims != 2:
            raise ValueError("Only 2 and 3 dimensional plots are supported")
        self._ax = ax
        self._dim = dims

        if animation_kws is not None:
            self._anime = animation.FuncAnimation(
                fig, self._animation_callback, **animation_kws
            )
        else:
            self._anime = animation.FuncAnimation(fig, self._animation_callback)

        if not self._ax.get_xlabel():
            self._ax.set_xlabel("X Position (m)")
        if not self._ax.get_ylabel():
            self._ax.set_ylabel("Y Position (m)")
        if dims == 3:
            assert isinstance(self._ax, Axes3D)

            if not self._ax.get_zlabel():
                self._ax.set_zlabel("Z Position (m)")

    @property
    def anime(self):
        return self._anime

    def set_data(self, x: ArrayLike, y: ArrayLike, z: Optional[ArrayLike] = None):
        """Sets the line data to be animated. The line data can be matrices,
        which will be treated as batches of 1D graph data stacked along the
        first axis. Plotting will then iterate across these batches, yielding
        multiple lines

        Parameters
        ----------
        x : ArrayLike
            Graph data along the X-coordinate
        y : ArrayLike
            Graph data along the Y-coordinate
        z : Optional[ArrayLike], optional
            Graph data along the Z-coordinate, if any, by default None
        """
        self.x = np.atleast_2d(np.asarray(x))
        self.y = np.atleast_2d(np.asarray(y))
        if z is not None and self._dim == 3:
            self.z = np.atleast_2d(np.asarray(z))

    def _animation_callback(self, _):
        self._ax.clear()

        if self._dim == 2:
            tup = self.x, self.y
        else:
            tup = self.x, self.y, self.z

        for plt_data in zip(*tup):
            self._ax.plot(*plt_data)
        self._ax.relim()
        self._ax.autoscale_view(True, True)

        if self.annotation and self._dim == 2:
            an_x = sum(self._ax.get_xlim()) / 2
            y_lower, y_upper = self._ax.get_ylim()
            an_y = float(y_lower + 0.8 * (y_upper - y_lower))
            self._ax.annotate(
                self.annotation, (an_x, an_y), horizontalalignment="center"
            )

        return ()
