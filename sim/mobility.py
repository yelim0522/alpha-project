"""Mobility models.

  - RandomWaypoint : independent users (low spatial/temporal correlation; herding
                     arises only by chance).
  - GroupFlow      : users travel in platoons along straight lanes (vehicles on a
                     road, a bus, a train). Members of a platoon cross the same
                     cell boundary within a few seconds of each other, which is
                     the regime where multi-user contention (herding) appears.

Real GPS traces (T-Drive, Rome/Porto taxi, nuScenes) can be plugged in by
replacing `step` with trace playback that sets user.x/y and velocity.
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


class GroupFlow:
    """Platoons of `group_size` users move together along straight lanes.

    Each platoon enters from a random edge, crosses the area, and re-enters from
    a new random edge on exit. Members are jittered by up to `spread` metres so
    they hand over within a short interval rather than at the same instant."""

    def __init__(self, width: float, height: float, speed: float, rng: random.Random,
                 group_size: int = 8, spread: float = 40.0):
        self.width = width
        self.height = height
        self.speed = speed
        self.rng = rng
        self.group_size = max(1, group_size)
        self.spread = spread
        self._groups = {}     # gid -> leader state dict(x, y, vx, vy, gen)
        self._offsets = {}    # uid -> (dx, dy)
        self._seen_gen = {}   # uid -> leader generation last applied to this user

    def _spawn_leader(self):
        side = self.rng.randrange(4)
        if side == 0:    # left -> right
            x, y, ang = 0.0, self.rng.uniform(0, self.height), 0.0
        elif side == 1:  # right -> left
            x, y, ang = self.width, self.rng.uniform(0, self.height), math.pi
        elif side == 2:  # bottom -> top
            x, y, ang = self.rng.uniform(0, self.width), 0.0, math.pi / 2
        else:            # top -> bottom
            x, y, ang = self.rng.uniform(0, self.width), self.height, -math.pi / 2
        ang += self.rng.uniform(-0.35, 0.35)
        return {"x": x, "y": y, "vx": self.speed * math.cos(ang),
                "vy": self.speed * math.sin(ang), "gen": 0}

    def _inside(self, g) -> bool:
        return (-self.spread <= g["x"] <= self.width + self.spread and
                -self.spread <= g["y"] <= self.height + self.spread)

    def step(self, user: User, dt: float) -> None:
        gid = user.id // self.group_size
        if gid not in self._groups:
            self._groups[gid] = self._spawn_leader()
        if user.id not in self._offsets:
            self._offsets[user.id] = (self.rng.uniform(-self.spread, self.spread),
                                      self.rng.uniform(-self.spread, self.spread))
        g = self._groups[gid]
        # The first member of each platoon (lowest id) advances the leader.
        if user.id == gid * self.group_size:
            g["x"] += g["vx"] * dt
            g["y"] += g["vy"] * dt
            if not self._inside(g):
                fresh = self._spawn_leader()
                fresh["gen"] = g["gen"] + 1
                self._groups[gid] = g = fresh
        # Re-entering from a new edge is a teleport, not a handover: flag it so the
        # simulator starts a fresh session instead of migrating state across the map.
        if self._seen_gen.get(user.id) != g["gen"]:
            user.respawned = self._seen_gen.get(user.id) is not None
            self._seen_gen[user.id] = g["gen"]
        dx, dy = self._offsets[user.id]
        user.vx, user.vy = g["vx"], g["vy"]
        user.x, user.y = g["x"] + dx, g["y"] + dy
