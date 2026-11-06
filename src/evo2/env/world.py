"""The Part 3 world: a square 2D arena with walls, food and agents (JAX).

Defined in docs/SPEC_part3.md, Section 3. All arrays have fixed sizes so that
many worlds can run at once with jax.vmap: agents live in `n_slots` slots with
an `alive` mask, and food lives in `max_food` slots of which the first
`n_food` are in use.

Order of events in one step (both modes):
  1. Actions are squashed into their ranges; alive agents turn, then move.
     Walls stop movement (positions are clipped to [0, L]).
  2. Eating: every present food item within an alive agent's reach (centre
     distance <= size) is eaten. If several agents reach one item, the agent
     with the lowest random draw this step gets it. An agent may eat several
     items in one step. Each item gives `food_energy`.
  3. Energy cost is paid by every agent alive at the start of the step.
  4. Death: energy <= 0, or (ecological mode only) age >= max_age.
  5. Births (ecological mode only): every alive agent with energy >=
     birth_threshold is a would-be parent. Parents are taken in a random order
     and matched to free slots in slot order; parents left over when the slots
     run out do not give birth and keep their energy.
  6. Food: eaten items regrow (probability rho each, at a new position drawn
     from the layout), then present items drift, then (every season_P steps)
     the layout centres and the uneaten items move to new places.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from flax import struct

LAYOUT_CODES = {"uniform": 0, "clusters": 1, "patches": 2}
TRAIT_NAMES = ("max_speed", "size", "sense_range")
N_TRAITS = 3


@dataclass(frozen=True)
class WorldConfig:
    """Constants that do not vary between environments (Python values)."""

    n_slots: int = 16
    max_food: int = 64
    k_max: int = 5
    n_sectors: int = 8
    trait_lo: tuple = (0.25, 0.5, 4.0)
    trait_hi: tuple = (2.0, 2.0, 20.0)
    default_traits: tuple = (1.0, 1.0, 10.0)
    max_turn: float = math.pi / 4
    base_cost: float = 0.5
    move_cost: float = 1.0
    sense_cost: float = 0.05
    food_energy: float = 40.0
    birth_energy: float = 100.0
    ecological: bool = False
    birth_threshold: float = 200.0
    parent_birth_cost: float = 120.0
    max_age: int = 1000
    birth_offset: float = 1.0
    episode_len: int = 500
    centre_margin: float = 0.15

    @classmethod
    def from_dict(cls, d: dict) -> "WorldConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown world settings: {sorted(unknown)}")
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        return cls(**clean)

    @property
    def obs_dim(self) -> int:
        return 3 * self.n_sectors + 1 + 2 + N_TRAITS


@struct.dataclass
class EnvParams:
    """Parameters of one environment (JAX scalars, so they can be batched)."""

    L: jax.Array          # arena side
    n_food: jax.Array     # food items when fully grown (int)
    layout: jax.Array     # 0 uniform, 1 clusters, 2 patches (int)
    k: jax.Array          # number of cluster or patch centres (int)
    sigma: jax.Array      # cluster spread
    patch_r: jax.Array    # patch radius
    rho: jax.Array        # regrowth probability per eaten item per step
    drift: jax.Array      # food drift distance per step
    season_P: jax.Array   # steps between seasonal moves (0 = never) (int)


def make_env_params(L, n_food, layout, k=2, sigma=5.0, patch_r=5.0, rho=0.03,
                    drift=0.0, season_P=0) -> EnvParams:
    """EnvParams from Python values; layout may be a name or a code."""
    if isinstance(layout, str):
        layout = LAYOUT_CODES[layout]
    return EnvParams(
        L=jnp.float32(L), n_food=jnp.int32(n_food), layout=jnp.int32(layout),
        k=jnp.int32(k), sigma=jnp.float32(sigma), patch_r=jnp.float32(patch_r),
        rho=jnp.float32(rho), drift=jnp.float32(drift), season_P=jnp.int32(season_P),
    )


@struct.dataclass
class WorldState:
    pos: jax.Array          # (N, 2)
    heading: jax.Array      # (N,)
    energy: jax.Array       # (N,)
    alive: jax.Array        # (N,) bool
    age: jax.Array          # (N,) int
    traits: jax.Array       # (N, 3): max speed, size, sense range
    prev_action: jax.Array  # (N, 2): squashed previous action, in [-1, 1]
    eaten: jax.Array        # (N,) int: food items eaten since birth / episode start
    food_pos: jax.Array     # (M, 2)
    food_present: jax.Array  # (M,) bool
    food_dir: jax.Array     # (M,) drift direction (radians)
    centres: jax.Array      # (K, 2)
    t: jax.Array            # step count (int)


class World:
    """Pure functions over WorldState; `cfg` is fixed when the object is made."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.trait_lo = jnp.asarray(cfg.trait_lo, jnp.float32)
        self.trait_hi = jnp.asarray(cfg.trait_hi, jnp.float32)
        self.default_traits = jnp.asarray(cfg.default_traits, jnp.float32)

    # ----- food layout -------------------------------------------------
    def sample_centres(self, key, params: EnvParams) -> jax.Array:
        m = self.cfg.centre_margin
        u = jax.random.uniform(key, (self.cfg.k_max, 2))
        return (m + (1.0 - 2.0 * m) * u) * params.L

    def sample_food_positions(self, key, params: EnvParams, centres, n: int) -> jax.Array:
        k_uni, k_idx, k_gauss, k_ang, k_rad = jax.random.split(key, 5)
        uni = jax.random.uniform(k_uni, (n, 2)) * params.L
        idx = jax.random.randint(k_idx, (n,), 0, params.k)
        c = centres[idx]
        gauss = c + params.sigma * jax.random.normal(k_gauss, (n, 2))
        ang = jax.random.uniform(k_ang, (n,)) * 2.0 * jnp.pi
        rad = params.patch_r * jnp.sqrt(jax.random.uniform(k_rad, (n,)))
        patch = c + rad[:, None] * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
        pos = jnp.where(params.layout == 0, uni, jnp.where(params.layout == 1, gauss, patch))
        return jnp.clip(pos, 0.0, params.L)

    def food_active(self, params: EnvParams) -> jax.Array:
        return jnp.arange(self.cfg.max_food) < params.n_food

    # ----- reset ---------------------------------------------------------
    def reset(self, key, params: EnvParams, traits: jax.Array, n_alive: int | jax.Array) -> WorldState:
        """New world: n_alive agents (slots 0..n_alive-1) at random places.

        traits: (n_slots, 3), clipped to bounds here.
        """
        cfg = self.cfg
        n, m = cfg.n_slots, cfg.max_food
        k_pos, k_head, k_cent, k_food, k_dir = jax.random.split(key, 5)
        alive = jnp.arange(n) < n_alive
        centres = self.sample_centres(k_cent, params)
        return WorldState(
            pos=jax.random.uniform(k_pos, (n, 2)) * params.L,
            heading=jax.random.uniform(k_head, (n,)) * 2.0 * jnp.pi,
            energy=jnp.where(alive, cfg.birth_energy, 0.0).astype(jnp.float32),
            alive=alive,
            age=jnp.zeros((n,), jnp.int32),
            traits=jnp.clip(traits.astype(jnp.float32), self.trait_lo, self.trait_hi),
            prev_action=jnp.zeros((n, 2), jnp.float32),
            eaten=jnp.zeros((n,), jnp.int32),
            food_pos=self.sample_food_positions(k_food, params, centres, m),
            food_present=self.food_active(params),
            food_dir=jax.random.uniform(k_dir, (m,)) * 2.0 * jnp.pi,
            centres=centres,
            t=jnp.int32(0),
        )

    # ----- sensors -------------------------------------------------------
    def _sector_min(self, d, ang, valid):
        """Nearest valid distance in each sector: (N, S); inf if none."""
        s = self.cfg.n_sectors
        width = 2.0 * jnp.pi / s
        sec = jnp.floor(jnp.mod(ang + width / 2.0, 2.0 * jnp.pi) / width).astype(jnp.int32)
        sec = jnp.clip(sec, 0, s - 1)
        hit = (sec[..., None] == jnp.arange(s)) & valid[..., None]
        return jnp.min(jnp.where(hit, d[..., None], jnp.inf), axis=1)

    def observe(self, state: WorldState, params: EnvParams) -> jax.Array:
        """Sensor readings for every slot: (N, obs_dim). Dead slots read 0."""
        cfg = self.cfg
        n, s = cfg.n_slots, cfg.n_sectors
        rng = state.traits[:, 2]

        def closeness(dmin):
            return jnp.where(dmin <= rng[:, None], 1.0 - dmin / rng[:, None], 0.0)

        # food
        rel = state.food_pos[None, :, :] - state.pos[:, None, :]
        d = jnp.linalg.norm(rel, axis=-1)
        ang = jnp.arctan2(rel[..., 1], rel[..., 0]) - state.heading[:, None]
        valid = state.food_present[None, :] & (d <= rng[:, None])
        food = closeness(self._sector_min(d, ang, valid))

        # other agents
        rel = state.pos[None, :, :] - state.pos[:, None, :]
        d = jnp.linalg.norm(rel, axis=-1)
        ang = jnp.arctan2(rel[..., 1], rel[..., 0]) - state.heading[:, None]
        valid = state.alive[None, :] & ~jnp.eye(n, dtype=bool) & (d <= rng[:, None])
        agents = closeness(self._sector_min(d, ang, valid))

        # walls, along each sector's centre line
        dirs = state.heading[:, None] + jnp.arange(s) * (2.0 * jnp.pi / s)
        dx, dy = jnp.cos(dirs), jnp.sin(dirs)
        x, y = state.pos[:, 0:1], state.pos[:, 1:2]
        eps = 1e-6
        tx = jnp.where(dx > eps, (params.L - x) / jnp.where(dx > eps, dx, 1.0),
                       jnp.where(dx < -eps, -x / jnp.where(dx < -eps, dx, 1.0), jnp.inf))
        ty = jnp.where(dy > eps, (params.L - y) / jnp.where(dy > eps, dy, 1.0),
                       jnp.where(dy < -eps, -y / jnp.where(dy < -eps, dy, 1.0), jnp.inf))
        walls = closeness(jnp.minimum(tx, ty))

        energy = jnp.clip(state.energy / cfg.birth_energy, 0.0, 2.0)[:, None]
        traits = (state.traits - self.trait_lo) / (self.trait_hi - self.trait_lo)
        obs = jnp.concatenate([food, agents, walls, energy, state.prev_action, traits], axis=1)
        return obs * state.alive[:, None]

    # ----- actions -------------------------------------------------------
    def squash(self, action: jax.Array, traits: jax.Array):
        """Raw outputs (N, 2) -> (turn, speed, squashed). Bounds always hold."""
        a = jnp.tanh(action)
        turn = self.cfg.max_turn * a[:, 0]
        speed = traits[:, 0] * (a[:, 1] + 1.0) / 2.0
        return turn, speed, a

    def step_cost(self, traits: jax.Array, speed: jax.Array) -> jax.Array:
        cfg = self.cfg
        size, sense = traits[:, 1], traits[:, 2]
        return cfg.base_cost + cfg.move_cost * size ** 3 * speed ** 2 + cfg.sense_cost * sense

    # ----- step ----------------------------------------------------------
    def step(self, key, state: WorldState, params: EnvParams, action: jax.Array):
        """One step. Returns (new_state, info). See the module docstring."""
        cfg = self.cfg
        n = cfg.n_slots
        k_eat, k_birth, k_regrow, k_season = jax.random.split(key, 4)
        alive0 = state.alive

        # 1. move
        turn, speed, a = self.squash(action, state.traits)
        turn = jnp.where(alive0, turn, 0.0)
        speed = jnp.where(alive0, speed, 0.0)
        heading = jnp.mod(state.heading + turn, 2.0 * jnp.pi)
        step_vec = speed[:, None] * jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=1)
        pos = jnp.clip(state.pos + step_vec, 0.0, params.L)

        # 2. eat
        d = jnp.linalg.norm(state.food_pos[None, :, :] - pos[:, None, :], axis=-1)
        cand = (d <= state.traits[:, 1:2]) & alive0[:, None] & state.food_present[None, :]
        draw = jax.random.uniform(k_eat, (n,))
        winner = jnp.argmin(jnp.where(cand, draw[:, None], jnp.inf), axis=0)
        food_eaten = jnp.any(cand, axis=0)
        won = (winner[None, :] == jnp.arange(n)[:, None]) & food_eaten[None, :]
        n_eat = jnp.sum(won, axis=1).astype(jnp.int32)
        gain = cfg.food_energy * n_eat

        # 3. pay
        cost = jnp.where(alive0, self.step_cost(state.traits, speed), 0.0)
        energy = state.energy + gain - cost
        age = state.age + alive0.astype(jnp.int32)

        # 4. die
        died = alive0 & (energy <= 0.0)
        if cfg.ecological:
            died = died | (alive0 & (age >= cfg.max_age))
        removed_at_death = jnp.sum(jnp.where(died, energy, 0.0))
        alive = alive0 & ~died
        energy = jnp.where(alive, energy, 0.0)

        new = state.replace(
            pos=pos, heading=heading, energy=energy, alive=alive, age=age,
            prev_action=jnp.where(alive0[:, None], a, 0.0),
            eaten=state.eaten + n_eat,
            food_present=state.food_present & ~food_eaten,
        )

        # 5. births
        parent_idx = jnp.full((n,), -1, jnp.int32)
        n_births = jnp.int32(0)
        if cfg.ecological:
            new, parent_idx, n_births = self._births(k_birth, new, params)

        # 6. food dynamics
        new = self._food_dynamics(k_regrow, k_season, new, params)
        new = new.replace(t=state.t + 1)

        info = {
            "n_eat": n_eat,
            "food_gain": jnp.sum(gain),
            "cost": jnp.sum(cost),
            "removed_at_death": removed_at_death,
            "died": died,
            "n_births": n_births,
            "birth_overhead": n_births * (cfg.parent_birth_cost - cfg.birth_energy),
            "parent_idx": parent_idx,
            "alive_start": alive0,
            "speed": speed,
        }
        return new, info

    def _births(self, key, state: WorldState, params: EnvParams):
        cfg = self.cfg
        n = cfg.n_slots
        k_order, k_place, k_head = jax.random.split(key, 3)
        eligible = state.alive & (state.energy >= cfg.birth_threshold)
        free = ~state.alive
        prio = jnp.where(eligible, jax.random.uniform(k_order, (n,)), jnp.inf)
        parent_order = jnp.argsort(prio)
        free_order = jnp.argsort(jnp.where(free, jnp.arange(n), n + jnp.arange(n)))
        n_births = jnp.minimum(jnp.sum(eligible), jnp.sum(free)).astype(jnp.int32)
        rank_ok = jnp.arange(n) < n_births
        slot = jnp.where(rank_ok, free_order, n)
        parent_idx = jnp.full((n,), -1, jnp.int32).at[slot].set(
            parent_order.astype(jnp.int32), mode="drop")
        paid = jnp.zeros((n,)).at[jnp.where(rank_ok, parent_order, n)].add(
            cfg.parent_birth_cost, mode="drop")
        is_child = parent_idx >= 0
        p = jnp.maximum(parent_idx, 0)

        ang = jax.random.uniform(k_place, (n,)) * 2.0 * jnp.pi
        offset = cfg.birth_offset * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
        child_pos = jnp.clip(state.pos[p] + offset, 0.0, params.L)
        child_head = jax.random.uniform(k_head, (n,)) * 2.0 * jnp.pi

        sel = is_child[:, None]
        state = state.replace(
            energy=jnp.where(is_child, cfg.birth_energy, state.energy - paid),
            alive=state.alive | is_child,
            pos=jnp.where(sel, child_pos, state.pos),
            heading=jnp.where(is_child, child_head, state.heading),
            age=jnp.where(is_child, 0, state.age),
            traits=jnp.where(sel, state.traits[p], state.traits),
            prev_action=jnp.where(sel, 0.0, state.prev_action),
            eaten=jnp.where(is_child, 0, state.eaten),
        )
        return state, parent_idx, n_births

    def _food_dynamics(self, k_regrow, k_season, state: WorldState, params: EnvParams):
        m = self.cfg.max_food
        active = self.food_active(params)
        k_u, k_p = jax.random.split(k_regrow)
        regrow = active & ~state.food_present & (jax.random.uniform(k_u, (m,)) < params.rho)
        new_pos = self.sample_food_positions(k_p, params, state.centres, m)
        food_pos = jnp.where(regrow[:, None], new_pos, state.food_pos)
        present = state.food_present | regrow

        # drift with bouncing off walls
        dx, dy = jnp.cos(state.food_dir), jnp.sin(state.food_dir)
        moved = food_pos + params.drift * jnp.stack([dx, dy], axis=1)
        over_x = (moved[:, 0] < 0.0) | (moved[:, 0] > params.L)
        over_y = (moved[:, 1] < 0.0) | (moved[:, 1] > params.L)
        mx = jnp.where(moved[:, 0] < 0.0, -moved[:, 0],
                       jnp.where(moved[:, 0] > params.L, 2.0 * params.L - moved[:, 0], moved[:, 0]))
        my = jnp.where(moved[:, 1] < 0.0, -moved[:, 1],
                       jnp.where(moved[:, 1] > params.L, 2.0 * params.L - moved[:, 1], moved[:, 1]))
        moved = jnp.clip(jnp.stack([mx, my], axis=1), 0.0, params.L)
        new_dir = jnp.arctan2(jnp.where(over_y, -dy, dy), jnp.where(over_x, -dx, dx))
        drifting = present & (params.drift > 0.0)
        food_pos = jnp.where(drifting[:, None], moved, food_pos)
        food_dir = jnp.where(drifting, new_dir, state.food_dir)

        # seasons
        k_c, k_f = jax.random.split(k_season)
        season = (params.season_P > 0) & (jnp.mod(state.t + 1, jnp.maximum(params.season_P, 1)) == 0)
        centres = jnp.where(season, self.sample_centres(k_c, params), state.centres)
        season_pos = self.sample_food_positions(k_f, params, centres, m)
        food_pos = jnp.where((season & present)[:, None], season_pos, food_pos)

        return state.replace(food_pos=food_pos, food_present=present,
                             food_dir=food_dir, centres=centres)

    # ----- outside spawning (OE population floor) --------------------------
    def spawn(self, key, state: WorldState, params: EnvParams, mask: jax.Array,
              traits: jax.Array) -> WorldState:
        """Bring the slots in `mask` to life at random places with birth
        energy and the given traits. Energy comes from outside the world;
        callers count it."""
        n = self.cfg.n_slots
        k_pos, k_head = jax.random.split(key)
        m2 = mask[:, None]
        return state.replace(
            pos=jnp.where(m2, jax.random.uniform(k_pos, (n, 2)) * params.L, state.pos),
            heading=jnp.where(mask, jax.random.uniform(k_head, (n,)) * 2.0 * jnp.pi, state.heading),
            energy=jnp.where(mask, self.cfg.birth_energy, state.energy),
            alive=state.alive | mask,
            age=jnp.where(mask, 0, state.age),
            traits=jnp.where(m2, jnp.clip(traits, self.trait_lo, self.trait_hi), state.traits),
            prev_action=jnp.where(m2, 0.0, state.prev_action),
            eaten=jnp.where(mask, 0, state.eaten),
        )

    # ----- environment change (OE) ----------------------------------------
    def change_env(self, key, state: WorldState, params: EnvParams) -> WorldState:
        """Move a running world to new parameters. Agents stay (clipped into
        the new arena); food is laid out afresh for the new parameters."""
        m = self.cfg.max_food
        k_c, k_f, k_d = jax.random.split(key, 3)
        centres = self.sample_centres(k_c, params)
        return state.replace(
            pos=jnp.clip(state.pos, 0.0, params.L),
            food_pos=self.sample_food_positions(k_f, params, centres, m),
            food_present=self.food_active(params),
            food_dir=jax.random.uniform(k_d, (m,)) * 2.0 * jnp.pi,
            centres=centres,
        )
