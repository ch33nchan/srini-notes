# Core RL Through comma Controls Challenge

Date started: 2026-06-28

This note is the bridge from openpilot model-runtime learning into core reinforcement learning.

The positioning goal:

```text
RL-first engineer
with AV full-stack understanding
using comma/openpilot/control problems as the applied domain
```

Repo:

[commaai/controls_challenge](https://github.com/commaai/controls_challenge)

## 1. Why This Challenge Matters

The challenge README says:

```text
Machine learning models can drive cars...
But they famously suck at doing low level controls.
Your goal is to write a good controller.
```

Code reference:

[README challenge statement](https://github.com/commaai/controls_challenge/blob/master/README.md#L17)

This sits exactly at the boundary we just learned in openpilot:

```text
openpilot model stack:
  camera/context/history -> desired motion target

controls challenge:
  desired lateral target -> steering command that tracks it
```

So the controls challenge is not about training the full openpilot vision model.

It is about:

```text
low-level control
tracking
rollout evaluation
policy/controller design
```

That makes it a good core RL playground.

## 2. The RL Frame

In reinforcement learning, we usually define:

```text
state / observation:
  what the agent sees

action:
  what the agent chooses

transition dynamics:
  how the world changes after the action

reward / cost:
  how good or bad the result was

policy:
  the decision rule mapping observation -> action

rollout / episode:
  one simulated trajectory over time
```

The controls challenge has all of these.

## 3. Environment Dynamics: TinyPhysics

The simulator is in:

[tinyphysics.py](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py)

The README describes TinyPhysics as:

```text
a simulated car trained to mimic lateral movement of a car given steering commands
```

Code reference:

[TinyPhysics README](https://github.com/commaai/controls_challenge/blob/master/README.md#L42)

TinyPhysics inputs:

```text
v_ego:
  vehicle speed

a_ego:
  forward acceleration

road_lataccel:
  lateral acceleration due to road roll

current_lataccel:
  current lateral acceleration

steer_action:
  steering command
```

TinyPhysics output:

```text
next/current lateral acceleration
```

So in RL terms:

```text
transition model:
  next_lataccel = f(history, state, steering_action)
```

It is autoregressive:

```text
new prediction depends on previous predicted lataccels too
```

Code reference:

[TinyPhysicsModel.get_current_lataccel](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L87)

```python
states = np.column_stack([actions, raw_states])
input_data = {
  'states': np.expand_dims(states, axis=0).astype(np.float32),
  'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
}
```

The learned simulator consumes recent:

```text
actions
vehicle states
past lateral acceleration predictions
```

and predicts the next lateral acceleration token.

## 4. State / Observation

The challenge defines:

[State and FuturePlan](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L40)

```python
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])
```

At each control step, the controller receives:

[controller update call](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L144)

```python
action = self.controller.update(
  self.target_lataccel_history[step_idx],
  self.current_lataccel,
  self.state_history[step_idx],
  future_plan=self.futureplan
)
```

So the controller sees:

```text
target_lataccel:
  desired lateral acceleration now

current_lataccel:
  current simulated lateral acceleration

state:
  roll_lataccel, v_ego, a_ego

future_plan:
  future target lataccel, roll_lataccel, v_ego, a_ego
```

In RL language, a first observation vector could be:

```text
[
  target_lataccel,
  current_lataccel,
  target_lataccel - current_lataccel,
  roll_lataccel,
  v_ego,
  a_ego,
  future target lataccel preview...
]
```

This is the observation design problem.

## 5. Action

The controller returns one value:

[BaseController API](https://github.com/commaai/controls_challenge/blob/master/controllers/__init__.py#L1)

```python
def update(self, target_lataccel, current_lataccel, state, future_plan):
  ...
  Returns:
    The control signal to be applied to the vehicle.
```

In the simulator:

[action clipping](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L144)

```python
action = self.controller.update(...)
action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
```

Constants:

[STEER_RANGE](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L33)

```python
STEER_RANGE = [-2, 2]
```

So the RL action is:

```text
continuous scalar steering command in [-2, 2]
```

This is a continuous-control problem.

Good candidate RL algorithms:

```text
PPO:
  robust first baseline, on-policy

SAC:
  strong continuous-control algorithm, off-policy

TD3:
  deterministic continuous-control baseline
```

Since the user wants PufferLib/PPO, start with PPO.

## 6. Reward / Cost

The challenge is framed as cost minimization.

Evaluation:

[README evaluation](https://github.com/commaai/controls_challenge/blob/master/README.md#L48)

```text
lataccel_cost:
  mean squared tracking error * 100

jerk_cost:
  mean squared lateral jerk * 100

total_cost:
  lataccel_cost * 50 + jerk_cost
```

Code reference:

[compute_cost](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L183)

```python
lat_accel_cost = np.mean((target - pred)**2) * 100
jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
```

In RL, we usually maximize reward.

So convert cost to reward:

```text
reward = -cost
```

A per-step reward could be:

```text
tracking_error = target_lataccel - current_lataccel
jerk = current_lataccel_t - current_lataccel_t_minus_1

reward_t =
  - tracking_error^2 * tracking_weight
  - jerk^2 * jerk_weight
  - action_change^2 * smoothness_weight
```

Important:

```text
The official leaderboard uses rollout-level cost.
RL training benefits from dense per-step rewards.
```

So the RL wrapper should expose dense rewards that approximate the final evaluation.

## 7. Rollout / Episode

A rollout happens here:

[TinyPhysicsSimulator.rollout](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L192)

```python
for _ in range(CONTEXT_LENGTH, len(self.data)):
  self.step()
return self.compute_cost()
```

One episode is:

```text
one route segment CSV
run controller from step 20 to the end
compute tracking and jerk cost
```

The first part of the segment is context:

[constants](https://github.com/commaai/controls_challenge/blob/master/tinyphysics.py#L27)

```python
FPS = 10
CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
```

Meaning:

```text
sim runs at 10 Hz
context/history length is 20 frames = 2 seconds
controller is evaluated after step 100
```

## 8. Baseline Policy: PID

Baseline controller:

[controllers/pid.py](https://github.com/commaai/controls_challenge/blob/master/controllers/pid.py)

```python
error = target_lataccel - current_lataccel
self.error_integral += error
error_diff = error - self.prev_error
return self.p * error + self.i * self.error_integral + self.d * error_diff
```

This is not RL.

It is a hand-coded policy:

```text
action = f(error, integral_error, derivative_error)
```

But it is crucial because RL should be compared against it.

Your RL project should start by reproducing PID results before training a policy.

## 9. Core RL Concepts Mapped To This Challenge

### MDP

The formal RL object:

```text
MDP = (S, A, P, R, gamma)
```

For controls challenge:

```text
S:
  simulator state/history + current target/future plan

A:
  steering command in [-2, 2]

P:
  TinyPhysics transition model

R:
  negative tracking/jerk/action-smoothness cost

gamma:
  discount factor, likely 0.99 or close
```

### Policy

Policy:

```text
pi(a | s)
```

Meaning:

```text
given observation/state,
produce action distribution
```

For PPO:

```text
policy outputs mean and std for continuous steering action
sample action during training
use mean or deterministic action during evaluation
```

### Value Function

Value:

```text
V(s) = expected future return from state s
```

PPO learns a value function to reduce variance and compute advantages.

For this challenge:

```text
V(s) estimates how well the controller can track from the current point onward.
```

### Advantage

Advantage:

```text
A(s, a) = how much better this action was than expected
```

PPO uses advantage estimates to update the policy.

In this environment:

```text
good action:
  reduces lateral acceleration error without creating jerk

bad action:
  reduces error too aggressively, overshoots, creates jerk, or destabilizes rollout
```

## 10. PPO Learning Loop

PPO loop:

```text
initialize policy

repeat:
  collect rollouts using current policy
  compute rewards
  estimate advantages
  update policy with clipped objective
  evaluate on held-out segments
```

In controls challenge terms:

```text
collect rollouts:
  run policy in TinyPhysics on many CSV segments

reward:
  negative tracking + jerk cost

policy update:
  improve steering actions that reduce future cost

evaluation:
  run official eval.py against PID baseline
```

## 11. Important RL Design Choices

### Observation Design

Minimum:

```text
target_lataccel
current_lataccel
error
roll_lataccel
v_ego
a_ego
```

Better:

```text
add short history of:
  error
  action
  current_lataccel

add future preview of:
  target_lataccel
  roll_lataccel
  v_ego
```

Why future preview matters:

```text
controllers should anticipate upcoming turns,
not only react after error appears.
```

### Action Design

Direct action:

```text
policy outputs steer_action
```

Residual action:

```text
policy outputs correction on top of PID/MPC
```

Residual is often easier:

```text
action = pid_action + rl_residual
```

This lets RL learn improvements rather than the whole controller from scratch.

### Reward Shaping

Avoid only final episode reward.

Use dense reward:

```text
- tracking error squared
- jerk squared
- action delta squared
- action magnitude penalty
```

But be careful:

```text
too much action penalty:
  policy becomes lazy and understeers

too little smoothness penalty:
  policy tracks but becomes jerky
```

### Train/Test Split

Do not optimize only the same segments.

Use:

```text
train segments
validation segments
official eval segments
```

This matters because a controller can overfit specific route dynamics.

## 12. Project Roadmap

### Stage 1: Understand And Reproduce

```text
clone controls_challenge
install requirements
run PID baseline on one segment
run PID baseline on 100 segments
read report.html
```

### Stage 2: Classical Controller

```text
implement PID + feedforward
add future target preview
tune with grid/random search
beat baseline PID
```

### Stage 3: Gym/PufferLib Environment

Create an environment with:

```text
reset(segment_id)
step(action)
observation vector
reward
done
info
```

The environment wraps:

```text
TinyPhysicsSimulator
```

### Stage 4: PPO

Train:

```text
policy network:
  observation -> steering action distribution

value network:
  observation -> expected future return
```

Evaluate:

```text
official eval.py
compare vs PID
report cost distribution
```

### Stage 5: Residual RL

Train policy to output:

```text
delta_action
```

Use:

```text
action = baseline_controller_action + delta_action
```

This is a strong engineering story:

```text
classical controller baseline
+ learned residual
+ rigorous evaluation
```

## 13. How To Present This Skillset

The story should not be:

```text
I trained PPO on a toy environment.
```

The stronger story:

```text
I studied openpilot's model-to-control boundary,
formalized comma's controls challenge as an RL problem,
built a Gym/PufferLib environment around a learned dynamics model,
reproduced classical baselines,
then trained and evaluated RL/residual-RL controllers against the official cost.
```

That says:

```text
RL fundamentals
AV control understanding
simulation literacy
evaluation discipline
production-code reading ability
```

That is the positioning.
