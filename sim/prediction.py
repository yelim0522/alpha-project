"""Mobility prediction shared by all proactive policies.

A constant-velocity-and-heading (CVH) extrapolation, the same class of predictor
Pallas evaluates with. It is deliberately simple so that the comparison between
policies isolates *how the prediction is used*, not the predictor itself.
Optional noise on speed/heading stresses prediction robustness.
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional

from environment import Server, User, distance, nearest_server_at


@dataclass
class Prediction:
    target: int      # predicted next serving server
    t_ho: float      # predicted handover time (absolute)
    margin: float    # confidence proxy in [0, 1]: how clearly the target wins


def predict(user: User, servers: List[Server], t: float, horizon: float, step: float,
            rng: Optional[random.Random] = None, speed_noise: float = 0.0,
            heading_noise: float = 0.0) -> Optional[Prediction]:
    speed = math.hypot(user.vx, user.vy)
    if speed <= 1e-9 or user.server < 0:
        return None
    vx, vy = user.vx, user.vy
    if rng is not None and (speed_noise > 0 or heading_noise > 0):
        scale = max(0.1, 1.0 + rng.gauss(0.0, speed_noise))
        ang = math.atan2(vy, vx) + rng.gauss(0.0, heading_noise)
        vx, vy = speed * scale * math.cos(ang), speed * scale * math.sin(ang)

    tau = step
    while tau <= horizon + 1e-9:
        px, py = user.x + vx * tau, user.y + vy * tau
        s = nearest_server_at(px, py, servers)
        if s.id != user.server:
            d_best, d_second = float("inf"), float("inf")
            for o in servers:
                d = distance(px, py, o.x, o.y)
                if d < d_best:
                    d_best, d_second = d, d_best
                elif d < d_second:
                    d_second = d
            margin = min(1.0, (d_second - d_best) / s.coverage) if d_second < float("inf") else 1.0
            return Prediction(s.id, t + tau, margin)
        tau += step
    return None


def predict_all(users: List[User], servers: List[Server], t: float, horizon: float,
                step: float, rng: Optional[random.Random] = None,
                speed_noise: float = 0.0, heading_noise: float = 0.0
                ) -> Dict[int, Prediction]:
    out = {}
    for u in users:
        p = predict(u, servers, t, horizon, step, rng, speed_noise, heading_noise)
        if p is not None:
            out[u.id] = p
    return out
