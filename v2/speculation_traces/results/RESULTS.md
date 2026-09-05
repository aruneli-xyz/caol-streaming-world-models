# Speculation frontier results

## Frozen trace evaluation

- Test set: 85 episodes, 28,310 eligible episode-local blocks; 21,397 are
  intent changes. One short test episode has no eligible post-history block.
- Train-only intent fit: yaw dead zone 11.100000
  degrees and pitch dead zone 4.950000 degrees,
  fitted over 125,864 blocks from 512 train episodes.
- Best intent result among the tested predictors is first-order Markov at
  B=4: 59.1381% overall (episode-bootstrap 95% CI
  56.7646%–61.4038%)
  and 47.3197% on intent changes
  (44.4826%–50.1101%).
- The frozen short-history Markov arm at B=4 reaches
  56.1780% overall and
  44.6979% on changes.
- Exact 12-frame sequence hashes remain sparse: the best B=4 tested values are
  9.6044% overall
  (7.5837%–11.7939%)
  and 2.0984% on changes
  (1.7437%–2.4700%).

## Measured Gamma systems replay

- Complete fixed-scene replay: 2 hash-frozen decisions × B={0,1,2,4}, serial
  on one H200 through the one-block delta path.
- Speculative candidate generation ranges from 4698.00 to
  4724.82 ms; the trace projection uses the recorded
  median 4725.17 ms.
- Measured charged GPU work across the eight scenarios is
  89.764 s. Every generated candidate
  and each miss fallback is included.
- With the declared 750 ms lead window, measured replay readiness is false for
  every speculative scenario. Simulated capacities C={1,2,4} therefore also
  project 0% trace-wide readiness; these are projections, not concurrent runs.
- Fork snapshots are zero bytes at the selected pre-roll boundary. Each captured
  candidate state is 4,965,556,224 bytes; B=4 retains 19,862,224,896 bytes on
  a hit path and 24,827,781,120 bytes when a miss fallback is also retained.
- This replay supports systems cost/readiness claims only. It uses one scene,
  two decisions, and train-derived representative intent branches; it does not
  measure semantic visual response. Paged copy-on-write allocator measurements
  are separate and were not substituted.
