"""Mobility models. A random-waypoint generator is provided; real GPS traces
(T-Drive, Rome/Porto taxi) can be plugged in by replacing `RandomWaypoint.step`
with trace playback that sets user.x/y and velocity per timestep.
"""

import math
import random

from environment import User


class RandomWaypoint:
    """Each user picks a random destination, moves toward it at a fixed speed,
    then picks a new destination on arrival."""

    def __init__(self, width: float, height: float, speed: float, rng: random.Random):
        self.width = width
        self.height = height
        self.speed = speed
        self.rng = rng
        self._targets = {}

    def _new_target(self):
        return (self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))

    def step(self, user: User, dt: float) -> None:
        if user.id not in self._targets:
            self._targets[user.id] = self._new_target()
        tx, ty = self._targets[user.id]
        dx, dy = tx - user.x, ty - user.y
        dist = math.hypot(dx, dy)
        stride = self.speed * dt
        if dist <= stride or dist == 0.0:
            user.x, user.y = tx, ty
            user.vx = user.vy = 0.0
            self._targets[user.id] = self._new_target()
        else:
            ux, uy = dx / dist, dy / dist
            user.vx, user.vy = ux * self.speed, uy * self.speed
            user.x += user.vx * dt
            user.y += user.vy * dt
