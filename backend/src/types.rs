use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[pyclass]
pub enum TyreCompound {
    Soft,
    Medium,
    Hard,
    Intermediate,
    Wet,
}

#[pymethods]
impl TyreCompound {
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
    fn __eq__(&self, other: &TyreCompound) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct DriverState {
    #[pyo3(get, set)]
    pub driver_id: String,
    #[pyo3(get, set)]
    pub position: u32,
    #[pyo3(get, set)]
    pub lap_number: u32,
    #[pyo3(get, set)]
    pub tyre_compound: TyreCompound,
    #[pyo3(get, set)]
    pub tyre_age: u32,
    #[pyo3(get, set)]
    pub gap_to_leader: f32,
    #[pyo3(get, set)]
    pub last_lap_time: f32,
    #[pyo3(get, set)]
    pub pit_stops: u32,
    #[pyo3(get, set)]
    pub status: String, // "OnTrack", "Pit", "Retired"
}

#[pymethods]
impl DriverState {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        driver_id: String,
        position: u32,
        lap_number: u32,
        tyre_compound: TyreCompound,
        tyre_age: u32,
        gap_to_leader: f32,
        last_lap_time: f32,
        pit_stops: u32,
        status: String,
    ) -> Self {
        Self {
            driver_id,
            position,
            lap_number,
            tyre_compound,
            tyre_age,
            gap_to_leader,
            last_lap_time,
            pit_stops,
            status,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct TyreConfig {
    #[pyo3(get, set)]
    pub degradation: f32,
    #[pyo3(get, set)]
    pub pace_offset: f32,
}

#[pymethods]
impl TyreConfig {
    #[new]
    fn new(degradation: f32, pace_offset: f32) -> Self {
        Self { degradation, pace_offset }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RaceConfig {
    #[pyo3(get, set)]
    pub total_laps: u32,
    #[pyo3(get, set)]
    pub track_length_km: f32,
    #[pyo3(get, set)]
    pub base_lap_time: f32,
    #[pyo3(get, set)]
    pub tyre_configs: HashMap<TyreCompound, TyreConfig>,
}

#[pymethods]
impl RaceConfig {
    #[new]
    fn new(
        total_laps: u32, 
        track_length_km: f32, 
        base_lap_time: f32,
        tyre_configs: HashMap<TyreCompound, TyreConfig>
    ) -> Self {
        Self {
            total_laps,
            track_length_km,
            base_lap_time,
            tyre_configs,
        }
    }
}
