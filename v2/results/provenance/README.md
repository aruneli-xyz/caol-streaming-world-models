# Gamma-World source provenance

`gamma_world_source.patch` is the exact binary-capable Git diff used by the
confirmatory generation runs. It is based on Gamma-World commit
`6a95de85c439d8ea73eae34c88fbfd4e89ea02e2`.

- Patch SHA-256:
  `814691ca7fbb8ca11633d9b5ea41bc3966a95ad1849049995b529e4ebdf8a75c`
- The hash matches `gamma_source.diff_sha256` in
  `../confirmatory/manifest.json`.
- No untracked files under the source paths included by `v2/preflight.py`
  were present when the patch was archived.

To reconstruct the source, check out the commit in a separate Gamma-World
worktree and inspect/apply the patch there. The experiment never applies this
patch automatically.
