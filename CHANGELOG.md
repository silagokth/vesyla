# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add changelog
- Add a new environment variable `VESYLA_SUITE_PATH_TMP` to store the temporary files. It is initialized in vs-entry. Note that, all temporary files are stored this directory should have some random name to avoid conflicts with other files.

### Fixed

### Changed

- Modify the vs-entry to read the version from the changelog to automatically determine the software version.
- Modify the vs-schedule to use VESYLA_SUITE_PATH_TMP to store the temporary files.

### Removed

- Remove the alimpsim sub-command

## [4.0.7] - 2023-04-01

### Added

### Fixed

### Changed

- Reorganize the drra-component library, the vs-component is modified to adapt to it.

### Removed

- Removed vs-alimpsim CMake file
