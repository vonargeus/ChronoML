"""Model package for training and artifact management. It contains scripts
that train baseline models, generate versioned artifacts, and capture metadata
needed for reproducibility. By isolating training logic from the API, we can
update models without rewriting HTTP code. This separation also supports later
additions like preprocessing, feature engineering, or model registries while
keeping inference stable and makes debugging much easier."""
