"""Write docs/PILOT_part3_P0.md from the saved stage P0 results (no numbers typed by hand).

  python3 scripts/part3/report_p0.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/part3/throughput"
# machine tag -> (label, $/hour actually charged, as returned by RunPod at creation)
MACHINES = {"rtx4090": ("RTX 4090 24 GB (secure)", 0.89), "a100sxm": ("A100 SXM 80 GB (community)", 1.39),
            "cpu5c32": ("CPU cpu5c, 32 vCPU (secure)", 1.12)}
METHODS = ["oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr"]
NAMES = {"oe": "OE", "oe_fixed": "OE-fixed", "ppo_fixed": "PPO-fixed", "ppo_dr": "PPO-DR", "es_dr": "ES-DR"}


def status_times(tag):
    t = {}
    for ln in (R / tag / "logs__status.txt").read_text().splitlines():
        ts, ev = ln.split(" ", 1)
        t.setdefault(ev, ts)
    return t


def minutes(a, b):
    from datetime import datetime
    f = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds() / 60


def main():
    s = {t: json.loads((R / t / "summary.json").read_text()) for t in MACHINES}
    out = []
    w = out.append
    w("# Part 3 pilot, stage P0: speed and repeatability (results)\n")
    w("We measured training throughput and within-machine repeatability on three hardware configurations.\n")
    w("## Machines\n")
    w("| Machine | Hardware recorded | Code received → setup done | Setup done → all done |")
    w("|---|---|---|---|")
    for t, (label, _) in MACHINES.items():
        h = s[t]["hardware"]
        st = status_times(t)
        hw = "; ".join(h.get("gpus") or []) or f"{h.get('cpu_count')} CPUs"
        hw += f"; JAX {h['jax']} ({h['backend']})" + (f"; CUDA driver {h['cuda_driver_version']}" if h.get("cuda_driver_version") else "")
        w(f"| {label} | {hw} | {minutes(st['code_received'], st['setup_ok']):.0f} min | "
          f"{minutes(st['setup_ok'], st['all_done']):.0f} min |")
    w("")
    w("- The CPU machine's recorded CPU count is the host's; the pod itself had 32 vCPU.")
    w("- The RTX 4090's host downloaded packages slowly, so its setup took most of its time.\n")
    w("## Repeatability\n")
    w("Each method was trained twice from the same seed (3090) on the same machine. "
      "\"Same\" means the two trained policies are bit-identical.\n")
    w("| Method | " + " | ".join(MACHINES[t][0] for t in MACHINES) + " |")
    w("|---|" + "---|" * len(MACHINES))
    for m in METHODS:
        w(f"| {NAMES[m]} | " + " | ".join("same" if s[t]["methods"][m]["repeat_bit_identical"] else "**differs**"
                                         for t in MACHINES) + " |")
    w("")
    w("## Training throughput\n")
    w("Living agent-steps per second, over full-size chunks after the first (compile time left out), "
      "two runs per method. "
      f"Run lengths: {s['rtx4090']['budget']:,} agent-steps for PPO and ES, {s['rtx4090']['oe_budget']:,} for OE.\n")
    w("| Method | " + " | ".join(f"{MACHINES[t][0]}: agent-steps/s" for t in MACHINES) + " |")
    w("|---|" + "---|" * len(MACHINES))
    for m in METHODS:
        cells = []
        for t, (_, price) in MACHINES.items():
            sp = [r["agent_steps_per_second"] for r in s[t]["methods"][m]["runs"]]
            cells.append(f"{sp[0]:,.0f} / {sp[1]:,.0f}")
        w(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    w("")
    w("## OE as several processes on one machine\n")
    w("k OE trainings started at once (seeds 3091–3098). \"Sum after compile\" adds each run's speed over its full "
      "chunks; those windows do not fully overlap, so it overstates the joint speed. \"Total with compile\" is all "
      "agent-steps divided by the wall time of the whole batch, including compilation.\n")
    w("| Machine | k | Sum after compile (agent-steps/s) | Total with compile (agent-steps/s) | Wall time |")
    w("|---|---|---|---|---|")
    for t, (label, _) in MACHINES.items():
        for c in s[t]["concurrent_oe"]:
            per = [x for x in c["per_run_agent_steps_per_second"] if x]
            w(f"| {label} | {c['processes']} | {sum(per):,.0f} | {c['total_agent_steps_per_second_incl_compile']:,.0f} "
              f"| {c['wall_seconds']:.0f} s |")
    w("")
    w("## Short runs are mostly compile time\n")
    w("| Machine | Method | Wall time of the 1st run | Time the same agent-steps take at full-chunk speed |")
    w("|---|---|---|---|")
    for t, (label, _) in MACHINES.items():
        for m in ("oe", "es_dr", "ppo_dr"):
            r = s[t]["methods"][m]["runs"][0]
            w(f"| {label} | {NAMES[m]} | {r['wall_seconds_total']:.0f} s | "
              f"{r['agent_steps'] / r['agent_steps_per_second']:.0f} s |")
    w("")
    (ROOT / "docs/PILOT_part3_P0.md").write_text("\n".join(out) + "\n")
    print("wrote docs/PILOT_part3_P0.md")


if __name__ == "__main__":
    main()
