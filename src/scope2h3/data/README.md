# Local data layout

This directory is intentionally excluded from Git except for this file and
`metadata.example.json`.

Training data can contain licensed videos, reference images, camera poses and
captions. Do not publish it by default. The smoke-test preparation script
creates `data/smoke_scene_sample_11/` locally from a source clip; it is also
ignored.

Use `metadata.example.json` as the portable schema reference. Every training
record needs a target video, reference input, content-only prompt, OpenCV c2w
camera pose array and horizontal field of view in radians.
