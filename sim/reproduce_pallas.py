"""Calibration check against the numbers published in Pallas (arXiv 2608.16477).

Part A  Table 1  : single handover, Qwen3-32B, 300 Mbps; Full-Copy / Recomputation /
                   ctHO at 1K, 2K, 4K tokens. Fitting set for (v1, T0_reactive).
Part B  Fig. 8a  : handover-handover contention, Qwen3-14B, 2,000-token context,
                   1 Gbps aggregate goodput, K = 1..4 UEs migrating to one target.
                   K = 1 fits T0_proactive; K = 4 worst-user fits activation_serial;
                   the remaining cells are validation.
Part C  Extrapolation beyond what two GPUs can measure: same setup, K = 6..32,
                   Pallas approx vs coordinated. This is where herding appears.

Run:  python3 reproduce_pallas.py [--t-avail 5] [--dt 0.1]
"""

import argparse

from environment import Server, MODEL_PRESETS, optimal_prefill_length, handover_delay
from minisim import mini_sim, params_for
from policies import PallasApprox, Coordinated

PALLAS_TABLE1 = {  # ms, Qwen3-32B @ 300 Mbps
    "Full-Copy":     {1000: 7165, 2000: 14310, 4000: 29066},
    "Recomputation": {1000: 581, 2000: 1208, 4000: 2103},
    "ctHO":          {1000: 554, 2000: 1119, 4000: 2058},
}
# Fig. 8a, Qwen3-14B, 2K context, 1 Gbps aggregate. Values read from the text /
# figure; None = not reported numerically.
PALLAS_FIG8A = {
    1: dict(pallas_avg=236, pallas_worst=237, ctho_worst=442),
    2: dict(pallas_avg=238, pallas_worst=None, ctho_worst=None),
    3: dict(pallas_avg=240, pallas_worst=283, ctho_worst=None),
    4: dict(pallas_avg=292, pallas_worst=359, ctho_worst=507),
}


def pct(sim, meas):
    return "   -  " if meas is None else f"{100.0 * (sim - meas) / meas:+5.0f}%"


# --------------------------------------------------------------------------- #
def part_a():
    model = "qwen32b"
    p, cp = MODEL_PRESETS[model], params_for(model)
    v1, c, B, T0 = p["prefill_speed"], cp.kv_mb_per_token, 300.0 / 8.0, cp.activation_latency
    print("Part A  Table 1 (Qwen3-32B, 300 Mbps, single handover)  [ms]  sim / measured / error")
    print(f"        fitted: v1={v1:.0f} tok/s, T0={T0 * 1000:.0f} ms, c={p['kv_kib']:.0f} KiB/token")
    for name in ("Full-Copy", "Recomputation", "ctHO"):
        cells = []
        for L in (1000, 2000, 4000):
            if name == "Full-Copy":
                d = c * L / B
            elif name == "Recomputation":
                d = L / v1
            else:
                pstar = optimal_prefill_length(L, c, v1, B)
                d = handover_delay(pstar, L - pstar, c, v1, B)
            sim = 1000.0 * (T0 + d)
            meas = PALLAS_TABLE1[name][L]
            cells.append(f"{sim:7.0f} / {meas:5d} / {pct(sim, meas)}")
        print(f"  {name:<14}" + "  |  ".join(cells))
    print()


# --------------------------------------------------------------------------- #
def hh_scenario(policy, K, model, link_mbps, context, t_avail, dt):
    """K UEs migrate simultaneously from server 0 to server 1. Returns list of SIT (s)."""
    return mini_sim(policy, model, link_mbps, [context] * K, [t_avail] * K, dt)


def ctho_hh(K, model, link_mbps, context):
    p, params = MODEL_PRESETS[model], params_for(model)
    srv = Server(1, 0.0, 0.0, 1.0, p["prefill_speed"], link_mbps / 8.0, 1.0)
    v, bw, c = srv.prefill_share(K), srv.backhaul_bw / K, params.kv_mb_per_token
    pstar = optimal_prefill_length(context, c, v, bw)
    return params.activation_latency + handover_delay(pstar, context - pstar, c, v, bw)


def part_b(t_avail, dt):
    model, link, ctx = "qwen14b", 1000.0, 2000
    p = MODEL_PRESETS[model]
    print("Part B  Fig. 8a (Qwen3-14B, 2K tokens, 1 Gbps aggregate, K concurrent UEs -> one target)  [ms]")
    print(f"        fitted: v1={p['prefill_speed']:.0f} tok/s, T0_react={p['t0_reactive'] * 1000:.0f} ms, "
          f"T0_proact={p['t0_proactive'] * 1000:.0f} ms, serial={40} ms, batched x4; T_avail={t_avail} s")
    print(f"  {'K':>2} | {'Pallas avg':>10} {'meas':>5} {'err':>6} | {'Pallas worst':>12} {'meas':>5} {'err':>6}"
          f" | {'ctHO worst':>10} {'meas':>5} {'err':>6}")
    for K in (1, 2, 3, 4):
        m = PALLAS_FIG8A[K]
        sits = hh_scenario(PallasApprox(), K, model, link, ctx, t_avail, dt)
        avg, worst = 1000 * sum(sits) / K, 1000 * max(sits)
        ct = 1000 * ctho_hh(K, model, link, ctx)
        print(f"  {K:>2} | {avg:10.0f} {m['pallas_avg'] or '-':>5} {pct(avg, m['pallas_avg'])}"
              f" | {worst:12.0f} {m['pallas_worst'] or '-':>5} {pct(worst, m['pallas_worst'])}"
              f" | {ct:10.0f} {m['ctho_worst'] or '-':>5} {pct(ct, m['ctho_worst'])}")
    print()


def part_c(t_avail, dt):
    model, link, ctx = "qwen14b", 1000.0, 2000
    print("Part C  Extrapolation: same setup, K beyond the prototype (no measured values)  [ms]")
    print(f"  {'K':>2} | {'ctHO worst':>10} | {'Pallas avg':>10} {'worst':>6} | {'Coord avg':>9} {'worst':>6} | "
          f"{'Pallas trig spread':>18} {'Coord trig spread':>17}")
    for K in (2, 4, 6, 8, 12, 16, 24, 32):
        pal, coo = PallasApprox(), Coordinated()
        s_p = hh_scenario(pal, K, model, link, ctx, t_avail, dt)
        s_c = hh_scenario(coo, K, model, link, ctx, t_avail, dt)
        ct = 1000 * ctho_hh(K, model, link, ctx)
        print(f"  {K:>2} | {ct:10.0f} | {1000 * sum(s_p) / K:10.0f} {1000 * max(s_p):6.0f}"
              f" | {1000 * sum(s_c) / K:9.0f} {1000 * max(s_c):6.0f} | "
              f"{_spread(pal):>18} {_spread(coo):>17}")
    print("\n  'trig spread' = time between the first and last preparation trigger (s); 0 = all fired together.")


def _spread(policy):
    tr = policy.trigger_log
    if not tr:
        return "-"
    return f"{max(tr) - min(tr):.2f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-avail", type=float, default=5.0, help="prediction lead before handover (s)")
    ap.add_argument("--dt", type=float, default=0.1, help="control period (s)")
    a = ap.parse_args()
    part_a()
    part_b(a.t_avail, a.dt)
    part_c(a.t_avail, a.dt)


if __name__ == "__main__":
    main()
