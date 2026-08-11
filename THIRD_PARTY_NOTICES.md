# Third-party notices

## Segment Anything Model 3

BiVSS-CD uses a modified fork of Meta's Segment Anything Model 3 (SAM3),
included as the `third_party/sam3` Git submodule. SAM3 code and model weights
are governed by the SAM License contained in that submodule, not by the
Apache-2.0 license of BiVSS-CD.

The fork is based on Meta's SAM3 repository and adds the interface required by
BiVSS-CD:

- `score_threshold_detection` and `new_det_thresh` are exposed through the
  video-predictor construction chain.
- `use_decoupled_selection` assigns newly detected objects to independent
  tracker states when enabled.

Pinned BiVSS-CD SAM3 revision:
`e4bc932fc5177a00ff65a99f37f0a7a698cf8c72`.

Upstream: https://github.com/facebookresearch/sam3
