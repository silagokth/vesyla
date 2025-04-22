# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add log_panics to vs-entry, vs-component, vs-manas and vs-testcase.

### Fixed

- Make sure vs-entry return non-zero exit code if the subcommand returns an error.
- Make subcommands return non-zero exit code if command line parsing fails.
- Make sure the -V and -h flag does not trigger error and return non-zero exit code.
- Fix the subcommand function call in vs-testcase.

### Changed

- Move logger initialization to the main function of vs-component.
- Change vs-entry to accept '-V' for version instead of '-v' in order to align with the clap
  convention. The '-v' flag is reserved for future verbose output.
- Change vs-testcase subcommand format.

### Removed

- Remove useless test in vs-component.

## [4.0.10] - 2025-04-17

### Fixed

- [vs-component] fixed error where the assembly would fail when overwriting custom_properties

## [4.0.9] - 2025-04-17

### Added

- Added version numbers for all tools [#32](https://github.com/silagokth/vesyla/issues/32)

### Fixed

- Fix SST template incompatibility with multi-col fabrics [#36](https://github.com/silagokth/vesyla/issues/36)

### Changed

- `--version` subcommand now also shows the versions of vesyla tools

## [4.0.8] - 2025-04-14

### Added

- Add changelog
- Add a new environment variable `VESYLA_SUITE_PATH_TMP` to store the temporary files. It is initialized in vs-entry. Note that, all temporary files are stored this directory should have some random name to avoid conflicts with other files.

### Fixed

- Fix bug: report error when component is not found in library.

### Changed

- Modify the vs-entry to read the version from the changelog to automatically determine the software version.
- Modify the vs-schedule to use VESYLA_SUITE_PATH_TMP to store the temporary files.

### Removed

- Remove the alimpsim sub-command

## [4.0.7] - 2025-04-01

### Added

### Fixed

### Changed

- Reorganize the drra-component library, the vs-component is modified to adapt to it.

### Removed

- Removed vs-alimpsim CMake file
