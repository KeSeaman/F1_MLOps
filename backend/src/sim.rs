//! Lap-by-lap race simulator: tyre wear, fuel burn, pit stops, drive modes and a simple
//! traffic model. Pure Rust; the Python bindings live in lib.rs.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::f64::consts::TAU;

#[derive(Debug, Clone, Deserialize)]
pub struct Compound {
    pub name: String,
    /// Seconds per lap relative to the fastest compound, on a fresh tyre.
    pub pace_offset: f64,
    /// Seconds lost per unit of wear (one normal lap = one unit).
    pub degradation: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct DriveMode {
    /// Seconds added to the lap (negative = faster).
    pub pace_delta: f64,
    /// Tyre wear for the lap, in normal-lap units.
    pub wear: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DriverSpec {
    pub id: String,
    /// Seconds per lap relative to the field average.
    pub pace_offset: f64,
    pub start_compound: String,
    /// Seconds behind the first car at the start.
    #[serde(default)]
    pub start_gap: f64,
    /// Wear already on the starting set (used sets from qualifying).
    #[serde(default)]
    pub start_wear: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SimConfig {
    pub total_laps: u32,
    pub base_lap_time: f64,
    /// Seconds per lap completed (negative: cars get faster as fuel burns).
    pub fuel_effect: f64,
    /// Time lost driving through the pit lane and stopping.
    pub pit_loss: f64,
    pub lap_noise_sd: f64,
    pub compounds: Vec<Compound>,
    pub push: DriveMode,
    pub save: DriveMode,
    /// A car must be this much faster over a lap to pass the car in front.
    pub overtake_delta: f64,
    /// Gap kept to the car in front when stuck behind it.
    pub follow_gap: f64,
    /// A car starting the lap within dirty_air_gap of the car in front loses dirty_air_loss.
    pub dirty_air_gap: f64,
    pub dirty_air_loss: f64,
    pub drivers: Vec<DriverSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Stay,
    Push,
    Save,
    /// Pit at the end of this lap for the compound at this index.
    Pit(usize),
}

#[derive(Debug, Clone)]
pub struct Driver {
    pub id: String,
    pub pace_offset: f64,
    pub compound: usize,
    /// Laps completed on the current set.
    pub tyre_age: u32,
    pub wear: f64,
    pub total_time: f64,
    pub last_lap_time: f64,
    pub pit_stops: u32,
    /// Bit i set when compound i has been used.
    pub compounds_used: u32,
    pub position: u32,
}

pub struct Race {
    pub config: SimConfig,
    pub drivers: Vec<Driver>,
    /// Driver indices in running order.
    pub order: Vec<usize>,
    /// Laps completed.
    pub lap: u32,
    rng: StdRng,
}

impl Race {
    pub fn new(config: SimConfig, seed: Option<u64>) -> Result<Self, String> {
        if config.total_laps == 0 {
            return Err("total_laps must be positive".into());
        }
        if config.compounds.is_empty() || config.compounds.len() > 32 {
            return Err("need between 1 and 32 compounds".into());
        }
        if config.drivers.is_empty() {
            return Err("need at least one driver".into());
        }
        if !(config.lap_noise_sd >= 0.0 && config.lap_noise_sd.is_finite()) {
            return Err("lap_noise_sd must be a non-negative number".into());
        }
        let mut drivers = Vec::with_capacity(config.drivers.len());
        for spec in &config.drivers {
            if drivers.iter().any(|d: &Driver| d.id == spec.id) {
                return Err(format!("duplicate driver id {:?}", spec.id));
            }
            let compound = compound_index(&config, &spec.start_compound)?;
            drivers.push(Driver {
                id: spec.id.clone(),
                pace_offset: spec.pace_offset,
                compound,
                tyre_age: spec.start_wear.round() as u32,
                wear: spec.start_wear,
                total_time: spec.start_gap,
                last_lap_time: 0.0,
                pit_stops: 0,
                compounds_used: 1 << compound,
                position: 0,
            });
        }
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let mut race = Self { order: (0..drivers.len()).collect(), config, drivers, lap: 0, rng };
        race.update_order();
        Ok(race)
    }

    pub fn finished(&self) -> bool {
        self.lap >= self.config.total_laps
    }

    /// Advance every driver one lap. `actions` is indexed like `drivers`.
    pub fn step(&mut self, actions: &[Action]) -> Result<(), String> {
        if self.finished() {
            return Err("the race is finished".into());
        }
        if actions.len() != self.drivers.len() {
            return Err(format!("expected {} actions, got {}", self.drivers.len(), actions.len()));
        }
        let cfg = &self.config;
        let mut lap_times = vec![0.0; self.drivers.len()];
        for (i, driver) in self.drivers.iter_mut().enumerate() {
            let (pace_delta, wear) = match actions[i] {
                Action::Push => (cfg.push.pace_delta, cfg.push.wear),
                Action::Save => (cfg.save.pace_delta, cfg.save.wear),
                Action::Stay | Action::Pit(_) => (0.0, 1.0),
            };
            driver.wear += wear;
            driver.tyre_age += 1;
            let compound = &cfg.compounds[driver.compound];
            let mut lap_time = cfg.base_lap_time
                + driver.pace_offset
                + compound.pace_offset
                + compound.degradation * driver.wear
                + cfg.fuel_effect * self.lap as f64
                + pace_delta
                + cfg.lap_noise_sd * gaussian(&mut self.rng);
            if let Action::Pit(next) = actions[i] {
                lap_time += cfg.pit_loss;
                driver.compound = next;
                driver.tyre_age = 0;
                driver.wear = 0.0;
                driver.pit_stops += 1;
                driver.compounds_used |= 1 << next;
            }
            lap_times[i] = lap_time;
        }

        // Traffic, in running order. A car close behind another loses time in its dirty air,
        // and one that would end the lap ahead of the car in front only gets past if it was at
        // least overtake_delta faster; otherwise it is held behind.
        let mut ahead: Option<(f64, f64)> = None; // (end time, lap time) of the previous car
        for &i in &self.order {
            let driver = &mut self.drivers[i];
            if let Some((ahead_end, ahead_lap)) = ahead {
                if driver.total_time - (ahead_end - ahead_lap) < cfg.dirty_air_gap {
                    lap_times[i] += cfg.dirty_air_loss;
                }
            }
            let mut end = driver.total_time + lap_times[i];
            if let Some((ahead_end, ahead_lap)) = ahead {
                if end < ahead_end + cfg.follow_gap && ahead_lap - lap_times[i] < cfg.overtake_delta {
                    end = ahead_end + cfg.follow_gap;
                }
            }
            driver.last_lap_time = end - driver.total_time;
            driver.total_time = end;
            ahead = Some((end, driver.last_lap_time));
        }

        self.lap += 1;
        self.update_order();
        Ok(())
    }

    pub fn leader_time(&self) -> f64 {
        self.drivers[self.order[0]].total_time
    }

    fn update_order(&mut self) {
        let drivers = &self.drivers;
        self.order.sort_by(|&a, &b| drivers[a].total_time.total_cmp(&drivers[b].total_time));
        for (pos, &i) in self.order.iter().enumerate() {
            self.drivers[i].position = pos as u32 + 1;
        }
    }
}

pub fn compound_index(config: &SimConfig, name: &str) -> Result<usize, String> {
    config
        .compounds
        .iter()
        .position(|c| c.name == name)
        .ok_or_else(|| format!("unknown compound {name:?}"))
}

/// Standard normal sample (Box-Muller).
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
