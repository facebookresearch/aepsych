# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.1] - 2022-05-18
### Changed
- Unity Version is now 2020.3.27f1

## [1.2.1] - 2022-05-05
### Added
- New option to query for next trial while current trial is underway
- Added public stimulusDuration float to Experiment Template
### Changed
- Improved Experiment Template messages

## [1.1.0] - 2022-04-12
### Added
- New public Experiment method: ShowOptimal() queries server for optimal parameters, then displays result to screen
- Time-out retry during initial server connection attempt

### Changed
- Experiment messages in Experiment Template to indicate initial connection status


## [1.0.0] - 2022-04-01
### Added
- Importable package formatting
- Automatic AEPsych server configuration from Unity editor
- Model Exploration for querying AEPsych model from Unity editor
- Experiment base class with built-in server communication methods
- Context Menu automated Experiment script generation
- Automatic CSV generation for experiment data
- New Audio 1D optimization demo scene

### Changed
- Moved from ZMQ to TCP server communication
- Updated example experiments to adhere to Experiment base class implementation
