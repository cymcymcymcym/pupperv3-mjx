
import jax
import jax.numpy as jp
import numpy as np
from pupperv3_mjx import environment
from pupperv3_mjx import config

# Setup environment params
xml_path = "test/test_pupper_model.xml"
reward_config = config.get_config()

env = environment.PupperV3Env(
    path=xml_path,
    reward_config=reward_config,
    action_scale=0.5,
    observation_history=15,
    environment_timestep=0.02,
    physics_timestep=0.004,
    force_probability=0.5, 
    force_duration_range=jp.array([10, 20]),
)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

print("Running Force Distribution Analysis...")
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
state = jit_reset(reset_rng)

force_vectors = []
n_steps = 5000 

for i in range(n_steps):
    action = jp.zeros(12)
    state = jit_step(state, action)
    
    # env._torso_idx is 1. 
    force = state.pipeline_state.xfrc_applied[env._torso_idx, 3:]
    
    if jp.linalg.norm(force) > 0.1:
        force_vectors.append(np.array(force))

force_vectors = np.array(force_vectors)

if len(force_vectors) == 0:
    print("No forces detected!")
else:
    print(f"Collected {len(force_vectors)} active force samples.")
    
    angles = np.arctan2(force_vectors[:, 1], force_vectors[:, 0])
    
    print(f"Mean Angle: {np.mean(angles):.3f} rad")
    print(f"Min Angle:  {np.min(angles):.3f} rad")
    print(f"Max Angle:  {np.max(angles):.3f} rad")
    
    q1 = np.sum((force_vectors[:,0] > 0) & (force_vectors[:,1] > 0))
    q2 = np.sum((force_vectors[:,0] < 0) & (force_vectors[:,1] > 0))
    q3 = np.sum((force_vectors[:,0] < 0) & (force_vectors[:,1] < 0))
    q4 = np.sum((force_vectors[:,0] > 0) & (force_vectors[:,1] < 0))
    
    print(f"Quadrant Distribution: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}")
    
    if q1 > 0 and q2 > 0 and q3 > 0 and q4 > 0:
        print("SUCCESS: Forces are distributed across all quadrants.")
    else:
        print("FAILURE: Forces are biased to specific quadrants!")
        
    z_vals = force_vectors[:, 2]
    print(f"Z Component Range: {np.min(z_vals):.3f} to {np.max(z_vals):.3f}")
    if np.any(z_vals < 0):
        print("WARNING: Found negative Z forces.")
