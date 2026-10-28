# Synthesizability lint — Tier A (open tools, no PDK)

The cheap, broad half of the synthesizability strategy, on the vesyla side.
Assembles a stdcell-only fabric with the freshly-built vesyla from the PR and
lints + elaborates it with **open tools only** (Verible + slang) on a stock
GitHub-hosted runner. No PDK, no liberty, no self-hosted server — runs on
**every PR**.

It mirrors the same Tier A gate in `drra-components`; that one guards RTL
changes, this one guards **vesyla** changes. Either can break the composed
fabric. It is the complement to the gated Genus gate (`docs/SYNTH_CI.md`), which
is the authoritative PDK area/Fmax sign-off.

## What it does

```
ci-synth-lint.yml (ubuntu-latest, every PR, after ci-build-vesyla)
  ├─ install Verible (required) + slang (best-effort prebuilt)
  ├─ download the drra-components `library` (branch artifact or release, via gh)
  ├─ install the vesyla AppImage built in THIS run (pkg-artifacts)
  ├─ vesyla component assemble -i scripts/synth/arch_smoke_no_sram.json -o build
  │     (pure RTL composition — no solver, no simulation, top module `fabric`)
  └─ scripts/synth/lint_fabric.sh build/rtl
```

`vesyla component assemble` is the bare composition path (no minizinc / no
QuestaSim), which is why this works on a vanilla runner. The input
`scripts/synth/arch_smoke_no_sram.json` is a minimal 1×1 **no-SRAM** fabric
(swb + io + 3×rf + dpu on `cell_single_row`), so it is stdcell-only and needs no
SRAM macro.

## What it catches

| Tool | Check | Severity |
|------|-------|----------|
| `verible-verilog-syntax` | parse breakage from templating / bad SV | hard fail |
| `verible-verilog-lint` | style + a few structural rules (`scripts/synth/verible.rules`) | soft (hard if `STRICT=1`) |
| `slang --top fabric` | full elaboration: unresolved modules, port/width mismatches, and — because the RTL uses `always_comb` — incomplete-assignment **latches** | hard fail (when slang present) |

slang natively elaborates the `agu_cfg_if` SystemVerilog interfaces that plain
Yosys cannot. It is best-effort: if no prebuilt binary is published, the step
skips with a notice and Verible remains the guaranteed baseline.

## Relationship to Tier B (Genus)

| | Tier A (this) | Tier B (`ci-synth.yml`) |
|--|--------------|--------------------------|
| Runner | GitHub-hosted | self-hosted w/ PDK server |
| PDK | none | GF22 (server-side only) |
| Catches | synth-construct breakage, latches | + real area / Fmax |
| Runs | every PR | gated (label + environment) |

## Local run

```sh
export VESYLA_SUITE_PATH_COMPONENTS=/path/to/built/library
vesyla component assemble -i scripts/synth/arch_smoke_no_sram.json -o build
bash scripts/synth/lint_fabric.sh build/rtl       # STRICT=1 to enforce lint too
```

## Notes / first-run

- The AppImage runs via `APPIMAGE_EXTRACT_AND_RUN=1` (no libfuse2 on the runner).
- If slang's prebuilt asset naming changes, adjust the grep in the install step.
- Uses existing secret `DRRA_COMPONENTS_PAT_VESYLA_READ_ONLY` for the library.
