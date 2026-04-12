# Simulation Pose Bookmarks

This feature lets you save and reuse robot poses in simulation so you can jump back to the same setup after restarting Gazebo or relaunching nodes.

## What It Does

- Teleport the robot to a saved pose by id
- Save the robot's current pose as a custom bookmark
- Keep a `latest` bookmark for quick retry loops

## Files

- Shared presets: `auv_sim/auv_sim_bridge/config/<namespace>/teleport_presets.yaml`
- Personal/custom poses: `auv_sim/auv_sim_bridge/config/<namespace>/teleport_custom.yaml`

`teleport_custom.yaml` is gitignored so each developer can keep their own local bookmarks.

## YAML Format

```yaml
gate_start:
  pose: {x: -7.0, y: -15.5, z: -0.6}
  rpy: {roll: 0.0, pitch: 0.0, yaw: 0.0}

torpedo_debug:
  pose: {x: 1.2, y: -3.4, z: -0.8}
  rpy: {roll: 0.0, pitch: 0.0, yaw: 1.57}
```

Angles are in radians.

## Services

- `/<namespace>/simulation/teleport_pose`
- `/<namespace>/simulation/save_pose`

Service type:

```srv
string pose_id
---
bool success
string message
string resolved_pose_id
```

## Usage

Save current pose as a named custom bookmark:

```bash
rosservice call /taluy/simulation/save_pose "pose_id: 'torpedo_debug'"
```

Save current pose as `latest`:

```bash
rosservice call /taluy/simulation/save_pose "pose_id: ''"
```

Teleport to a named pose:

```bash
rosservice call /taluy/simulation/teleport_pose "pose_id: 'torpedo_debug'"
```

Teleport to `latest`:

```bash
rosservice call /taluy/simulation/teleport_pose "pose_id: ''"
```

## Recommended Workflow

- Put team-wide task setups in `teleport_presets.yaml`
- Use `save_pose` for your own temporary debugging spots
- Leave `pose_id` empty when you just want to retry the last position quickly

## Notes

- Preset ids are read-only from the save service
- Saving a named custom pose also updates `latest`
- Teleport resets model velocity and best-effort syncs controller command pose afterward
