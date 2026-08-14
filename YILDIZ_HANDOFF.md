# Vitpose node services (all resolve under `/taluy`)

| Service | Type | Does |
|---|---|---|
| `vitpose_detection_node/set_mode` | `auv_msgs/SetString` | `off` \| `detect` \| `pose`. Loaded ≡ enabled: `detect` = objectness boxes only (joint model NOT on the GPU; bbox topic capped ~5 Hz), `pose` = boxes + `VitposeResult`, `off` = everything unloaded. Sync + truthful (a load takes ~1.3 s); `pose` fails on a detect-only config |
| `vitpose_detection_node/enable` | `std_srvs/SetBool` | alias: true = fullest mode the config supports (`pose`, or `detect` for detect-only), false = `off`. Re-enable after off pays the load again |
| `vitpose_detection_node/set_config` | `auv_msgs/SetString` | switch object: `gate` \| `tetra` (or a YAML path); keeps the current mode (a detect-only config clamps `pose`→`detect`) |
| `vitpose_scan_node/*` | — | same node type, same services (its config is detect-only: modes `off`/`detect`) |

Status, latched 1 Hz on both nodes: `.../enabled` (`std_msgs/Bool`, true = mode ≠ off) + `.../mode` (`std_msgs/String`).

Sim twin mirrors the whole interface.

```bash
rosservice call /taluy/vitpose_detection_node/set_mode "data: 'detect'"
rosservice call /taluy/vitpose_detection_node/enable "data: true"
rosservice call /taluy/vitpose_detection_node/set_config "data: 'tetra'"
rosservice call /taluy/vitpose_scan_node/enable "data: true"
```
