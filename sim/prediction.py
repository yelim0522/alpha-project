"""Mobility prediction shared by all proactive policies.

A constant-velocity-and-heading (CVH) extrapolation, the same class of predictor
Pallas evaluates with. It is deliberately simple so that the comparison between
policies isolates *how the prediction is used*, not the predictor itself.
Optional noise on speed/heading stresses prediction robustness.
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple

from environment import Server, User, distance, nearest_server_at


@dataclass(frozen=True)
class Candidate:
    """One possible next cell from an ensemble mobility prediction."""
    target: int
    t_ho: float
    probability: float


@dataclass
class Prediction:
    target: int      # predicted next serving server
    t_ho: float      # predicted handover time (absolute)
    margin: float    # confidence proxy in [0, 1]: how clearly the target wins
    candidates: Tuple[Candidate, ...] = ()


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
                speed_noise: float = 0.0, heading_noise: float = 0.0,
                candidate_count: int = 1, candidate_samples: int = 1
                ) -> Dict[int, Prediction]:
    out = {}
    for u in users:
        if candidate_count > 1 and candidate_samples > 1:
            p = predict_candidates(u, servers, t, horizon, step, rng, speed_noise,
                                   heading_noise, candidate_count, candidate_samples)
        else:
            p = predict(u, servers, t, horizon, step, rng, speed_noise, heading_noise)
        if p is not None:
            out[u.id] = p
    return out


def predict_candidates(user: User, servers: List[Server], t: float, horizon: float,
                       step: float, rng: Optional[random.Random], speed_noise: float,
                       heading_noise: float, candidate_count: int = 2,
                       samples: int = 16) -> Optional[Prediction]:
    """Return the top next-cell hypotheses from a noisy trajectory ensemble.

    Probability is the fraction of *all* samples voting for a target, so missing
    handovers inside the horizon retain their uncertainty mass instead of being
    silently renormalised away.
    """
    if speed_noise <= 0 and heading_noise <= 0:
        p = predict(user, servers, t, horizon, step, rng, 0.0, 0.0)
        if p is None:
            return None
        only = Candidate(p.target, p.t_ho, 1.0)
        return Prediction(p.target, p.t_ho, 1.0, (only,))
    votes: Dict[int, List[Prediction]] = {}
    for _ in range(max(1, samples)):
        p = predict(user, servers, t, horizon, step, rng, speed_noise, heading_noise)
        if p is not None:
            votes.setdefault(p.target, []).append(p)
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda item: (-len(item[1]), item[0]))
    candidates = []
    for target, preds in ranked[:max(1, candidate_count)]:
        candidates.append(Candidate(target, sum(p.t_ho for p in preds) / len(preds),
                                    len(preds) / max(1, samples)))
    primary = candidates[0]
    return Prediction(primary.target, primary.t_ho, primary.probability, tuple(candidates))
