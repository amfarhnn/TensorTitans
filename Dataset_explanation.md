## What The Project Solves

Imagine a wildfire has just been detected. During its **first five hours**, several measurements are collected.

Using only those early measurements, the model must answer:

> **What is the probability that this wildfire will come within 5 km of an evacuation-zone center within 12, 24, 48, or 72 hours?**

Each row represents **one wildfire event**.

---

## Important Concepts

### Fire Perimeter

A fire perimeter is the estimated boundary surrounding the burning area.

If the perimeter expands rapidly, the fire is growing quickly.

### Fire Centroid

The centroid is the approximate center point of the fire perimeter.

Tracking the centroid shows the direction and speed at which the overall fire is moving.

### Evacuation-Zone Centroid

This is the center point of an evacuation zone.

The fire is considered a threat when it comes within **5 km** of this point.

### `0_5h`

Columns ending with `0_5h` use measurements collected during the fire’s **first five hours**.

---

# Column Explanations

## Identifiers And Targets

| Column | Plain-language meaning |
|---|---|
| `event_id` | Anonymous unique identifier for the wildfire. It has no predictive meaning and should not be used as a model feature. |
| `time_to_hit_hours` | Hours after the first five-hour observation period until the fire came within 5 km of an evacuation zone. For fires that never reached it, this stores the final observation time. |
| `event` | Final result. `1` means the fire came within 5 km during the 72-hour window. `0` means it did not. |

`time_to_hit_hours` and `event` are only available in `train.csv` because these are what the model must learn to predict.

---

## Measurement Quality

These columns describe how much early fire-perimeter information was available.

| Column | Meaning |
|---|---|
| `num_perimeters_0_5h` | Number of fire-boundary measurements recorded during the first five hours. |
| `dt_first_last_0_5h` | Hours between the first and last available perimeter measurements. |
| `low_temporal_resolution_0_5h` | `1` means the early measurements are limited or too close together. Predictions based on these rows may be less reliable. |

For example, measuring a fire only once provides much less information about its growth direction than measuring it four times.

---

## Fire Size And Growth

These columns describe how large the fire is and how quickly it expands.

| Column | Meaning |
|---|---|
| `area_first_ha` | Fire area at the first observation, measured in hectares. One hectare is approximately the size of a large sports field. |
| `area_growth_abs_0_5h` | Total number of hectares added during the first five hours. |
| `area_growth_rel_0_5h` | Growth relative to the fire’s original size. |
| `area_growth_rate_ha_per_h` | Average hectares added per hour. |
| `log1p_area_first` | Log-transformed initial area. This reduces the influence of extremely large fires. |
| `log1p_growth` | Log-transformed absolute growth. |
| `log_area_ratio_0_5h` | Logarithmic comparison between final and initial fire area. |
| `relative_growth_0_5h` | Another representation of growth relative to initial size. |
| `radial_growth_m` | Approximate increase in the fire’s radius, measured in meters. |
| `radial_growth_rate_m_per_h` | Approximate radius expansion per hour. |

### Example

A fire growing from 10 to 20 hectares:

- Absolute growth: `10 hectares`
- Relative growth: `100%`
- A model may consider this dangerous because the fire doubled in size quickly.

---

## Fire-Center Movement

These columns describe how the fire’s center moves.

| Column | Meaning |
|---|---|
| `centroid_displacement_m` | Distance the fire’s center moved during the first five hours. |
| `centroid_speed_m_per_h` | Average movement speed of the fire’s center. |
| `spread_bearing_deg` | Direction the fire moved, measured in degrees. Approximately `0° = north`, `90° = east`, `180° = south`, and `270° = west`. |
| `spread_bearing_sin` | Mathematical representation of the movement direction using sine. |
| `spread_bearing_cos` | Mathematical representation of the movement direction using cosine. |

Sine and cosine are used because normal degree values can confuse models. Directions `359°` and `1°` are almost identical, although their raw numbers appear far apart.

---

## Distance To Evacuation Zone

Here, `ci` refers to the nearest evacuation-zone centroid.

| Column | Meaning |
|---|---|
| `dist_min_ci_0_5h` | Smallest distance between the fire and nearest evacuation-zone center during the first five hours. |
| `dist_std_ci_0_5h` | Variation in the recorded distances. High values indicate inconsistent distance changes. |
| `dist_change_ci_0_5h` | Difference between the fire’s first and last recorded distance from the evacuation zone. |
| `dist_slope_ci_0_5h` | Overall trend in distance over time. It indicates whether the fire is moving closer or farther away. |
| `closing_speed_m_per_h` | Speed at which the fire approaches the evacuation zone. Positive values mean it is moving closer. |
| `closing_speed_abs_m_per_h` | Magnitude of the closing speed, ignoring whether the fire moves closer or farther away. |
| `projected_advance_m` | Estimated distance the fire advances toward the evacuation zone. |
| `dist_accel_m_per_h2` | Change in closing speed. It indicates whether the fire is approaching increasingly quickly. |
| `dist_fit_r2_0_5h` | Confidence in the calculated distance trend. Values closer to `1` mean the trend is more consistent. |

These are likely among the most important predictors.

A fire that is already close and rapidly approaching an evacuation zone is considerably more threatening than a distant fire moving away.

---

## Direction Toward Evacuation Zone

These columns determine whether the fire is moving in the direction of the evacuation zone.

| Column | Meaning |
|---|---|
| `alignment_cos` | Directional alignment between fire movement and the evacuation zone. Positive values mean movement toward it; negative values generally mean movement away. |
| `alignment_abs` | Strength of the alignment regardless of whether it is toward or away. Values closer to `1` indicate stronger alignment. |
| `cross_track_component` | Sideways movement relative to the direction of the evacuation zone. |
| `along_track_speed` | Movement speed directly along the path toward or away from the evacuation zone. |

### Simple Movement Illustration

```text
Evacuation zone
      ↑
      ↑  Along-track movement
      ↑
Fire  →  Cross-track movement
```

High positive along-track speed is concerning because the fire is moving directly toward the evacuation zone.

---

## Start-Time Information

| Column | Meaning |
|---|---|
| `event_start_hour` | Hour when the fire started, from `0` to `23`. |
| `event_start_dayofweek` | Day the fire started: `0 = Monday`, through `6 = Sunday`. |
| `event_start_month` | Month the fire started: `1 = January`, through `12 = December`. |

These can indirectly represent environmental or response conditions.

For example:

- Certain months may be hotter and drier.
- Nighttime fires may be detected or managed differently.
- Seasonal conditions may affect fire growth.

---

# What The Model Should Learn

The model should combine several signals:

```text
Current distance
+ Fire growth speed
+ Movement direction
+ Movement speed
+ Measurement reliability
+ Seasonal information
= Probability of threatening an evacuation zone
```

A high-risk example might be:

- already close to an evacuation zone;
- rapidly increasing in size;
- moving quickly;
- moving directly toward the evacuation zone; and
- accelerating toward it.

A low-risk example might be large but very distant and moving away from communities.

The key challenge is not simply predicting **whether** a wildfire becomes dangerous. The model must also predict **how soon** it could become dangerous.