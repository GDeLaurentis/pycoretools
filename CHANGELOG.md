# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated


## [0.1.1] - 2026-01-07

### Changed

- Default cores for mapThreads is now min(available cores / 4, 16)


## [0.1.0] - 2026-01-07

### Added

- concurrency: mapThreads, filterThreads
- context: TemporarySetting
- decorators: retry, with_cm
- iterables: flatten, crease, chunks, all_non_empty_subsets
- sentinels: NaI


[unreleased]: https://github.com/GDeLaurentis/pycoretools/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/GDeLaurentis/pycoretools/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GDeLaurentis/pycoretools/releases/tag/v0.1.0
