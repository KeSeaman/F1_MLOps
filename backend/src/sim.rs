use std::collections::HashMap;
use crate::types::{DriverState, TyreCompound, RaceConfig, TyreConfig};
use rand::Rng;

pub struct RaceSim {
    pub drivers: HashMap<String, DriverState>,
    pub config: RaceConfig,
    pub current_lap: u32,
}

impl RaceSim {
    pub fn new(config: RaceConfig, drivers: Vec<DriverState>) -> Self {
        let mut driver_map = HashMap::new();
        for d in drivers {
            driver_map.insert(d.driver_id.clone(), d);
        }
        Self {
            drivers: driver_map,
            config,
            current_lap: 0,
        }
    }

    pub fn step(&mut self, actions: HashMap<String, String>) -> HashMap<String, DriverState> {
        // actions: driver_id -> "PIT_SOFT", "PIT_MEDIUM", "PUSH", "SAVE", "StayOut"
        
        self.current_lap += 1;
        let mut rng = rand::thread_rng();

        for (id, driver) in self.drivers.iter_mut() {
            let action = actions.get(id).map(|s| s.as_str()).unwrap_or("StayOut");
            
            // Handle Pit Stops
            if action.starts_with("PIT_") {
                driver.pit_stops += 1;
                driver.tyre_age = 0;
                driver.status = "OnTrack".to_string(); 
                match action {
                    "PIT_SOFT" => driver.tyre_compound = TyreCompound::Soft,
                    "PIT_MEDIUM" => driver.tyre_compound = TyreCompound::Medium,
                    "PIT_HARD" => driver.tyre_compound = TyreCompound::Hard,
                    "PIT_INTER" => driver.tyre_compound = TyreCompound::Intermediate,
                    "PIT_WET" => driver.tyre_compound = TyreCompound::Wet,
                    _ => {},
                }
                // Pit loss penalty (approx 20-25s)
                driver.gap_to_leader += 22.0; 
            } else {
                driver.tyre_age += 1;
            }

            // Calculate Lap Time using Config
            let default_config = TyreConfig { degradation: 0.1, pace_offset: 0.0 };
            let tyre_config = self.config.tyre_configs.get(&driver.tyre_compound).unwrap_or(&default_config);
            
            let deg_penalty = (driver.tyre_age as f32) * tyre_config.degradation;
            let pace_offset = tyre_config.pace_offset;
            let random_var = rng.gen_range(-0.2..0.2);
            
            let lap_time = self.config.base_lap_time + pace_offset + deg_penalty + random_var;
            driver.last_lap_time = lap_time;
            
            // Update gap (simplified, relative to a virtual leader doing base time)
            // In reality, we need to re-calculate gaps based on all drivers' times
            // For now, just accumulate time
        }

        // Re-sort positions based on total race time (not tracked here yet, need to add)
        // For now, just return state
        self.drivers.clone()
    }
}
