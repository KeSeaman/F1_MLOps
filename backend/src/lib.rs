mod sim;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use sim::{compound_index, Action, SimConfig};

/// Snapshot of one driver after the last completed lap.
#[pyclass(module = "f1_mlops._sim", frozen, get_all)]
#[derive(Clone)]
struct DriverState {
    id: String,
    position: u32,
    compound: String,
    tyre_age: u32,
    wear: f64,
    total_time: f64,
    gap_to_leader: f64,
    last_lap_time: f64,
    pit_stops: u32,
    /// Number of different compounds used so far.
    compounds_used: u32,
}

#[pymethods]
impl DriverState {
    fn __repr__(&self) -> String {
        format!(
            "DriverState(id={:?}, position={}, compound={:?}, tyre_age={}, gap_to_leader={:.3})",
            self.id, self.position, self.compound, self.tyre_age, self.gap_to_leader
        )
    }
}

/// A race between the drivers in a JSON config (see sim.rs `SimConfig`).
#[pyclass(module = "f1_mlops._sim")]
struct Race {
    inner: sim::Race,
}

#[pymethods]
impl Race {
    #[new]
    #[pyo3(signature = (config_json, seed=None))]
    fn new(config_json: &str, seed: Option<u64>) -> PyResult<Self> {
        let config: SimConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid sim config: {e}")))?;
        let inner = sim::Race::new(config, seed).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Advance one lap. `actions` maps driver id to "STAY", "PUSH", "SAVE" or
    /// "PIT_<COMPOUND>"; drivers without an entry stay out.
    fn step(&mut self, actions: HashMap<String, String>) -> PyResult<()> {
        let mut parsed = vec![Action::Stay; self.inner.drivers.len()];
        for (id, action) in &actions {
            let i = self.index(id)?;
            parsed[i] = match action.as_str() {
                "STAY" => Action::Stay,
                "PUSH" => Action::Push,
                "SAVE" => Action::Save,
                other => match other.strip_prefix("PIT_") {
                    Some(name) => Action::Pit(
                        compound_index(&self.inner.config, name).map_err(PyValueError::new_err)?,
                    ),
                    None => return Err(PyValueError::new_err(format!("unknown action {other:?}"))),
                },
            };
        }
        self.inner.step(&parsed).map_err(PyValueError::new_err)
    }

    fn driver(&self, id: &str) -> PyResult<DriverState> {
        Ok(self.state(self.index(id)?))
    }

    /// All drivers in running order.
    fn standings(&self) -> Vec<DriverState> {
        self.inner.order.iter().map(|&i| self.state(i)).collect()
    }

    /// Laps completed.
    #[getter]
    fn lap(&self) -> u32 {
        self.inner.lap
    }

    #[getter]
    fn total_laps(&self) -> u32 {
        self.inner.config.total_laps
    }

    #[getter]
    fn finished(&self) -> bool {
        self.inner.finished()
    }

    #[getter]
    fn compounds(&self) -> Vec<String> {
        self.inner.config.compounds.iter().map(|c| c.name.clone()).collect()
    }
}

impl Race {
    fn index(&self, id: &str) -> PyResult<usize> {
        self.inner
            .drivers
            .iter()
            .position(|d| d.id == id)
            .ok_or_else(|| PyKeyError::new_err(id.to_string()))
    }

    fn state(&self, i: usize) -> DriverState {
        let d = &self.inner.drivers[i];
        DriverState {
            id: d.id.clone(),
            position: d.position,
            compound: self.inner.config.compounds[d.compound].name.clone(),
            tyre_age: d.tyre_age,
            wear: d.wear,
            total_time: d.total_time,
            gap_to_leader: d.total_time - self.inner.leader_time(),
            last_lap_time: d.last_lap_time,
            pit_stops: d.pit_stops,
            compounds_used: d.compounds_used.count_ones(),
        }
    }
}

#[pymodule]
fn _sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DriverState>()?;
    m.add_class::<Race>()?;
    Ok(())
}
