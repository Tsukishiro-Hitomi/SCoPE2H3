# Architecture

`diffsynth` owns generic H3 loading, diffusion execution, VAE processing, and
the packed-token model pipeline. `scope2h3` owns every Ref2VA/SCoPE-specific
concern: camera trajectories, Pluecker encoding, attention injection, adapter
weights, and train/inference commands.

The only supported extension boundary is `install_scope_h3(pipe)` followed by
`set_scope_camera(pipe, poses, x_fov)`. The base pipeline remains usable
without SCoPE, and omitting a camera condition must remain an exact no-op.
