"""Named random streams for Part 2 (docs/PREREG_part2.md, Sections 2 and 10).

Every random number in a Part 2 run comes from a stream named by a key:
(purpose id, then integers). The stream is numpy's SeedSequence with the run
seed as entropy and the key as spawn_key, so different keys give independent
streams and the same key always gives the same numbers.

PURPOSES lists every purpose the code may use. A key always starts with its
purpose id, so two purposes can never share a stream by accident. Streams that
are shared on purpose (Section 10) are shared by using the *same* key.
"""
import numpy as np

# purpose name -> id. Never renumber: ids are part of every saved result.
PURPOSES = {
    "search_A1": 1,       # A search stage 1: (sub, [gen])
    "search_A2": 2,       # A search stage 2: (set, sub, [gen])
    "search_B1": 3,       # B search stage 1: (world id, sub, [gen])
    "search_B2": 4,       # B search stage 2: (world id, set, sub, [gen])
    "u2": 5,              # U2 check runs: (world id, set, sub, [gen])
    "tiebreak_A": 6,      # one key per design, A search
    "tiebreak_B": 7,      # one key per design, B search: (world id,)
    "deploy": 8,          # deployment: (phase, sub, [gen]); shared by all groups and B worlds
    "sv_designs": 9,      # SV_frozen's 50 random designs
    "clone_pick": 10,     # ET_cloneB's randomly picked blob: (mutation setting id,)
    "acheck": 11,         # A-checks for rows 5 and 6: (sub, [gen])
    "landscape": 12,      # landscape check, B stage 1 per suite variant: (world id, sub, [gen])
}
# The arm 3 map-size rule (seeds 2800-2829) uses the "deploy" streams of its own seeds.

# sub-stream ids inside a purpose
FOOD, BEHAVIOR, SPOTS = 0, 1, 2
PHASE_ID = {"A": 0, "B": 1}

_used = {}
_context = ["none"]
_log = None  # set to a list to record (key, context) for the stream-key test


class context:
    """Label the current use of streams (e.g. 'deploy_B:main:ET'), so the
    stream-key test can check that each key serves only its listed purposes."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        _context.append(self.label)
        return self

    def __exit__(self, *exc):
        _context.pop()
        return False


def start_logging():
    global _log
    _log = []


def stop_logging():
    global _log
    out, _log = _log, None
    return out


def seq(run_seed, purpose, *key):
    """SeedSequence for (purpose, *key) of one run seed."""
    if purpose not in PURPOSES:
        raise KeyError(f"unknown stream purpose {purpose!r}")
    k = (PURPOSES[purpose],) + tuple(int(x) for x in key)
    _used.setdefault(k[:1], set()).add(purpose)
    if _log is not None:
        _log.append((k, _context[-1]))
    return np.random.SeedSequence(int(run_seed), spawn_key=k)


def rng(run_seed, purpose, *key):
    return np.random.default_rng(seq(run_seed, purpose, *key))


def food_rng_for_gen(run_seed, purpose, *key):
    """A function g -> Generator giving each generation its own food stream."""
    return lambda g: rng(run_seed, purpose, *key, FOOD, g + 1)  # g = -1 (fixed centers) -> 0


def used_purposes():
    """For the stream-key test: the purpose names seen for each purpose id."""
    return {k: set(v) for k, v in _used.items()}
