# Third-party provenance

This directory is an experimental overlay for the pinned
[Mammoth](https://github.com/aimagelab/mammoth) checkout at commit
`e75a491c69fd729edeb01431afb753d9157d9a81`.

The ER, A-GEM, MER, and DER wrappers retain Mammoth's method structure and are
distributed under the upstream MIT license. The GEM wrapper retains the GEM
projection structure from `models/gem.py`; that upstream file identifies the
separate `gem_license` (Creative Commons Attribution-NonCommercial 4.0). The
launcher verifies the pinned checkout and its license bytes before installing
this overlay. This repository does not remove or replace either upstream
license.

`mer_current_only.py` is an experiment-local no-memory control that retains
MER's one-example stream batch and inner/outer meta-update schedule.

`pcgrad_replay_pairing.py` is a prespecified two-objective replay adaptation
inspired by PCGrad. It is deliberately reported as **PCGrad replay adaptation**,
not as an official continual-learning implementation supplied by the PCGrad
authors.
