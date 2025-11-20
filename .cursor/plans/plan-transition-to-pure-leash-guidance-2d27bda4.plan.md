<!-- 2d27bda4-c778-4482-a085-784553e639ef 558b6e17-eb9d-48ba-9656-1e868cacd49e -->
# Plan: Transition to Pure Leash Guidance

## Overview

Run the policy without external velocity commands and redesign the reward to emphasize following the moving leash target while keeping gait stability.

## Steps

1. Command Input

- Comment out command sampling/resampling sections, forcing `state.info['command'] = jp.zeros(3)` in `reset` and `step`.
- Keep observation layout unchanged so command slots remain but stay zeroed.

2. Reward Adjustments

- Add new reward terms in `rewards.py`:
- `reward_leash_compliance`: Penalize stretch `max(0, dist - slack)^2` or reward exponential falloff.
- `reward_leash_direction`: Encourage velocity alignment with leash direction (normalized dot product).
- `reward_leash_speed_match`: Encourage matching target speed along leash axis.
- Integrate these into reward config with appropriate weights; comment out or zero legacy command tracking rewards.

3. Stabilizing Terms

- Retain existing torque, orientation, slip penalties.
- Optionally retune weights to balance compliance vs. stability.

4. Testing

- Run zero-action leash compliance notebook to verify force behavior with new rewards.
- Observe whether the agent moves toward/leads the leash target smoothly.
- Iterate reward weights if oscillations or stalling occur.