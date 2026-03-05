# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated


## [0.1.1] - 2026-03-05

### Changed

- Default cores `default_cores()` for `mapThreads` is now `min(os.cpu_count() / 4, 16)`
- Multiprocessing picklability checks using ForkingPickler (supports functools.partial correctly)
- Reimplement Pool._repopulate_pool_static to create non-daemonic workers, preserving stdlib behaviour while avoiding daemon-related failures
- More tests 

### Fixed

- Overall, fix Python 3.11 SemLock context errors and CUDA + multiprocessing incompatibilities, keeping behaviour stable on Python 3.10
- Use explicit multiprocessing contexts instead of mutating the global start method
- Ensure Locks and Pools are created from the same context (fork/spawn)
- Fixed pool shutdown semantics
- Fixed issue where dict subclasses were being treated as namespaces in `TemporarySetting`
- Fixed badges links


## [0.1.0] - 2026-01-07

### Added

- concurrency: `mapThreads`, `filterThreads`
- context: `TemporarySetting`
- decorators: `retry`, `with_cm`
- iterables: `flatten`, `crease`, `chunks`, `all_non_empty_subsets`
- sentinels: `NaI`


[unreleased]: https://github.com/GDeLaurentis/pycoretools/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/GDeLaurentis/pycoretools/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GDeLaurentis/pycoretools/releases/tag/v0.1.0
