from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String, UnixTimestamp

# Define an entity for the driver
driver = Entity(name="driver", join_keys=["Driver"])

# Define the file source
laps_source = FileSource(
    name="laps_source",
    path="/home/captc/Devwork/F1_MLOps/data/laps.parquet",
    timestamp_field="EventTimestamp",
)

# Feature View
driver_stats_fv = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="TyreLife", dtype=Float32),
        Field(name="GapToLeader", dtype=Float32),
        Field(name="Position", dtype=Int64),
        Field(name="LapNumber", dtype=Int64),
        Field(name="Compound", dtype=String),
    ],
    online=True,
    source=laps_source,
)
