# Synchronizing DiffSynth

Treat `src/diffsynth` as a controlled vendor copy. Record the DiffSynth commit
in `NOTICE.md`, review its H3-specific changes, and run the full test suite
before and after each sync. Project-specific logic must stay in `src/scope2h3`;
add a generic hook to `diffsynth` only when no adapter-level extension can work.
