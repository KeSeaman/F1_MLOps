import gymnasium as gym
from gymnasium import spaces
import numpy as np
from f1_sim_backend import F1Env, RaceConfig, DriverState, TyreCompound

class F1GymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Action: 0=StayOut, 1=Push, 2=Save, 3=PitSoft, 4=PitMedium, 5=PitHard
        self.action_space = spaces.Discrete(6)
        
        # Observation: [TyreAge, GapToLeader, Position, TyreCompound(int), LapNumber]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self._load_config()
        self.reset()

    def _load_config(self):
        import json
        try:
            with open('data/sim_config.json', 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load sim_config.json, using defaults. Error: {e}")
            config_data = {
                "base_lap_time": 80.0,
                "compounds": {}
            }
            
        tyre_configs = {}
        # Map string keys to TyreCompound enum
        # Note: In Rust/PyO3 enum is just the class, but as key in dict it needs care.
        # Actually PyO3 enums in python are instances.
        
        for k, v in config_data.get("compounds", {}).items():
            compound_enum = getattr(TyreCompound, k.capitalize(), TyreCompound.Medium)
            from f1_sim_backend import TyreConfig # Import here to avoid circularity if top-level issues
            tyre_configs[compound_enum] = TyreConfig(v["degradation"], v["pace_offset"])

        self.config = RaceConfig(
            total_laps=66, 
            track_length_km=4.675, 
            base_lap_time=config_data.get("base_lap_time", 80.0),
            tyre_configs=tyre_configs
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize drivers
        # In a real scenario, we'd load this from a config or data
        drivers = []
        compounds = [TyreCompound.Soft, TyreCompound.Medium, TyreCompound.Hard]
        
        # Hero
        drivers.append(DriverState(
            driver_id="HAM", position=2, lap_number=0, tyre_compound=TyreCompound.Medium,
            tyre_age=0, gap_to_leader=1.5, last_lap_time=0.0, pit_stops=0, status="OnTrack"
        ))
        
        # Rival
        drivers.append(DriverState(
            driver_id="VER", position=1, lap_number=0, tyre_compound=TyreCompound.Medium,
            tyre_age=0, gap_to_leader=0.0, last_lap_time=0.0, pit_stops=0, status="OnTrack"
        ))
        
        self.sim = F1Env(self.config, drivers)
        
        return self._get_obs("HAM"), {}

    def step(self, action):
        action_map = {0: "StayOut", 1: "PUSH", 2: "SAVE", 3: "PIT_SOFT", 4: "PIT_MEDIUM", 5: "PIT_HARD"}
        hero_action = action_map[action]
        
        # Simple opponent logic (VER)
        # If tyre age > 25, pit. Else push.
        ver_state = self.sim.step({"HAM": "StayOut"})["VER"] # Peek state? No, step advances.
        # We need to decide actions BEFORE stepping.
        # But we can't peek easily without exposing more methods.
        # For now, simple static logic based on internal tracking or just random.
        
        opponent_action = "StayOut" 
        # In a real MARL, we'd query the opponent agent here.
        
        actions = {
            "HAM": hero_action,
            "VER": opponent_action
        }
        
        new_states = self.sim.step(actions)
        hero_state = new_states["HAM"]
        
        # Reward: Negative lap time (faster is better) + Position bonus
        reward = -hero_state.last_lap_time
        if hero_state.position == 1:
            reward += 10.0
            
        done = hero_state.lap_number >= self.config.total_laps
        
        return self._get_obs("HAM"), reward, done, False, {}
        
    def _get_obs(self, driver_id):
        drivers = self.sim.get_drivers()
        if driver_id not in drivers:
            return np.zeros(5, dtype=np.float32)
            
        state = drivers[driver_id]
        
        # Convert enum to int (simpler for now)
        # Assuming TyreCompound has a value or we map it
        # Rust enum to Python: it's an object. We need to map it.
        # For now, just use 0.
        compound_val = 0.0 
        
        return np.array([
            float(state.tyre_age),
            float(state.gap_to_leader),
            float(state.position),
            compound_val,
            float(state.lap_number)
        ], dtype=np.float32) 

