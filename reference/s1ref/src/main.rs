// Native reference harness for the MinuteLabs evolution simulator core.
//
// Source: https://github.com/minutelabsio/evolution-simulator, commit 55af956,
// by Jasper Palfree, GPL-3.0. The simulation modules below are compiled
// unchanged from that source tree via #[path]. This file only replaces the
// WebAssembly entry point (api/square_world.rs) with a command-line one that
// does the same steps and prints per-generation statistics as JSON lines.
//
// Usage: s1ref --seed N --gens G [--size 500] [--food 50] [--count 50]
//              [--trait-sd 0.5] [--reach-sd 0] [--no-cannibalism]

#[macro_use]
extern crate serde_derive;
extern crate nalgebra as na;

#[path = "../../minutelabs-evolution-simulator/src/wasm/src/stage/mod.rs"]
pub mod stage;
#[path = "../../minutelabs-evolution-simulator/src/wasm/src/creature/mod.rs"]
pub mod creature;
#[path = "../../minutelabs-evolution-simulator/src/wasm/src/simulation/mod.rs"]
pub mod simulation;
#[path = "../../minutelabs-evolution-simulator/src/wasm/src/math/mod.rs"]
mod math;
pub use math::*;

use creature::Creature;
use simulation::{behaviours, Simulation};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match args.iter().position(|a| a == name) {
        Some(i) => args[i + 1].parse().ok().expect("bad argument value"),
        None => default,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = arg(&args, "--seed", 0);
    let gens: u32 = arg(&args, "--gens", 100);
    let size: f64 = arg(&args, "--size", 500.0);
    let food: f64 = arg(&args, "--food", 50.0);
    let count: usize = arg(&args, "--count", 50);
    let trait_sd: f64 = arg(&args, "--trait-sd", 0.5);
    let reach_sd: f64 = arg(&args, "--reach-sd", 0.0);
    let cannibalism = !args.iter().any(|a| a == "--no-cannibalism");

    // Same creature template the web app builds (src/store/simulation.js:18-27, 110-126).
    let template_json = format!(
        r#"{{"id":"00000000000000000000000000000000","species":"default","state":"ACTIVE",
            "speed":[10.0,{sd}],"size":[10.0,{sd}],"sense_range":[20.0,{sd}],"reach":[1.0,{rsd}],
            "flee_distance":[1e12,0.0],"life_span":[1e4,0.0],"energy":500.0,
            "foods_eaten":[],"energy_consumed":0.0,"age":0,"pos":[0.0,0.0],"home_pos":[0.0,0.0],
            "movement_history":[[0.0,0.0]],"status_history":[],"objective":null}}"#,
        sd = trait_sd,
        rsd = reach_sd
    );
    let template: Creature = serde_json::from_str(&template_json).expect("template");

    // Same steps as SquareWorld::new / use_preset("default") / add_creatures / run
    // (api/square_world.rs:11-46, api/mod.rs:26-61), minus the wasm bindings.
    let stage = Box::new(stage::SquareStage(size));
    let mut sim = Simulation::new(stage, seed, Interpolator::new(&vec![(0.0, food)]));
    sim.add_behaviour(Box::new(behaviours::BasicMoveBehaviour));
    sim.add_behaviour(Box::new(behaviours::WanderBehaviour));
    if cannibalism {
        sim.add_behaviour(Box::new(behaviours::CannibalismBehaviour { size_ratio: 0.8 }));
    }
    sim.add_behaviour(Box::new(behaviours::ScavengeBehaviour));
    sim.add_behaviour(Box::new(behaviours::SatisfiedBehaviour));
    sim.add_behaviour(Box::new(behaviours::EdgeHomeBehaviour { disabled_edges: vec![] }));
    sim.add_behaviour(Box::new(behaviours::StarveBehaviour));

    let mut creatures = vec![];
    for _ in 0..count {
        let pos = sim.stage.get_nearest_edge_point(&sim.get_random_location());
        creatures.push(template.with_new_position(&pos));
    }
    sim.run(creatures, gens);

    for (gi, g) in sim.generations.iter().enumerate() {
        let n = g.creatures.len();
        let mut food_balls = 0usize;
        let mut prey = 0usize;
        let mut survivors = 0usize;
        let mut births = 0usize;
        let mut creature_steps = 0usize;
        let mut sums = [0.0f64; 4];
        let mut sq = [0.0f64; 4];
        let mut food_hist = [0usize; 4]; // creatures with 0,1,2,3+ food
        for c in &g.creatures {
            for e in &c.foods_eaten {
                if e.2 == "food_ball" { food_balls += 1 } else { prey += 1 }
            }
            let k = c.foods_eaten.len().min(3);
            food_hist[k] += 1;
            if c.is_alive() {
                survivors += 1;
                if c.foods_eaten.len() > 1 { births += 1 }
            }
            creature_steps += c.movement_history.len() - 1;
            let t = c.get_traits();
            let v = [t["speed"].0, t["size"].0, t["sense_range"].0, t["reach"].0];
            for i in 0..4 { sums[i] += v[i]; sq[i] += v[i] * v[i]; }
        }
        let nf = n as f64;
        let mean: Vec<f64> = (0..4).map(|i| sums[i] / nf).collect();
        let var: Vec<f64> = (0..4).map(|i| (sq[i] / nf - (sums[i] / nf).powi(2)).max(0.0)).collect();
        println!(
            "{{\"gen\":{},\"population\":{},\"food_balls_eaten\":{},\"prey_eaten\":{},\"survivors\":{},\"births\":{},\"ticks\":{},\"creature_steps\":{},\"food_hist\":[{},{},{},{}],\"mean\":{{\"speed\":{},\"size\":{},\"sense\":{},\"reach\":{}}},\"var\":{{\"speed\":{},\"size\":{},\"sense\":{},\"reach\":{}}}}}",
            gi, n, food_balls, prey, survivors, births, g.steps - 1, creature_steps,
            food_hist[0], food_hist[1], food_hist[2], food_hist[3],
            mean[0], mean[1], mean[2], mean[3], var[0], var[1], var[2], var[3]
        );
    }
}
