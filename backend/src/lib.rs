mod types;
mod sim;

use pyo3::prelude::*;
use std::collections::HashMap;
use types::{DriverState, RaceConfig, TyreCompound, TyreConfig};
use sim::RaceSim;

#[pyclass]
struct F1Env {
    sim: RaceSim,
}

#[pymethods]
impl F1Env {
    #[new]
    fn new(config: RaceConfig, drivers: Vec<DriverState>) -> Self {
        let sim = RaceSim::new(config, drivers);
        F1Env { sim }
    }

    fn step(&mut self, actions: HashMap<String, String>) -> HashMap<String, DriverState> {
        self.sim.step(actions)
    }
    
    fn get_drivers(&self) -> HashMap<String, DriverState> {
        self.sim.drivers.clone()
    }
    
    #[getter]
    fn current_lap(&self) -> u32 {
        self.sim.current_lap
    }
}

#[pymodule]
fn f1_sim_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TyreCompound>()?;
    m.add_class::<DriverState>()?;
    m.add_class::<TyreConfig>()?;
    m.add_class::<RaceConfig>()?;
    m.add_class::<F1Env>()?;
    Ok(())
}
