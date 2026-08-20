# The Apertus v1.5 stack branch (`apertus-stack`)

Apertus-v1.5-70B (swiss-ai) requires forked dependencies that our normal
lock cannot provide — this branch swaps them **in the lockfile**, so the
driver's runtime `uv sync`s preserve the swap instead of reverting it.
The driver itself is untouched; image and branch ship together as usual.

## What this branch changes

- `pyproject.toml`:
  - `transformers` → `git+https://github.com/swiss-ai/transformers.git
    @3797303dda74844e3d1f8977ff5518bb91f818b4` (via `[tool.uv.sources]`;
    the three version-range requirements are loosened to name-only so the
    fork's version string cannot conflict — the git rev IS the pin).
  - `vllm` → `git+https://github.com/swiss-ai/vllm.git
    @a601a9d998ddeb488f0c17e8512874b116aa7658` (the stock 0.25.1 wheel
    URLs are replaced by a name-only dep; **source build**).
- `uv.lock`: ⚠️ **NOT yet regenerated** — see the procedure below.

## Relock procedure (needs network + the container's torch)

The vLLM fork builds from source and its build backend needs torch at
metadata time, so run the relock **inside the current image** (which has
the toolchain) on a machine with network, e.g. an interactive container
on a login/compute node:

```bash
git switch apertus-stack
UV_OFFLINE=0 uv lock          # resolves the forks; vllm metadata build takes a while
git add uv.lock && git commit -s -m "chore(apertus): relock with swiss-ai forks"
git push fork apertus-stack
```

Then build the image:

```bash
podman build ... --build-arg NEMORL_COMMIT=apertus-stack ...
# expect a LONG build: the swiss vllm compiles its CUDA kernels from source
```

And run with:

```bash
NEMORL_BRANCH=apertus-stack sbatch train-gsm8k-apertus70b-nemorl.sh
```

## Known risks / open items (in expected order of appearance)

1. **Resolution conflicts at `uv lock`**: other dependencies pin
   transformers ranges (e.g. megatron-bridge, modelopt); the fork's
   version string may fall outside them.  Fix by loosening the offending
   constraint on this branch — same pattern as the three already done.
   **HIT (first image build, 2026-08-19)**: sglang==0.5.12.post1 pins
   `transformers==5.6.0` exactly; with the source pin the fork
   (5.14.0.dev0) is the only transformers in existence → the sglang
   split is unsatisfiable on every platform (the error names the x86_64
   split only because the resolver tried it first).  Fixed by adding
   the fork to `override-dependencies` (the timm precedent) — an
   override replaces ALL transitive transformers constraints.  Expect
   repeats of this shape for any other exact `transformers==` pin;
   same fix.
   ⚠️ Note on that build's log: it failed at STEP 40 (venv prefetch),
   but every earlier `uv sync --frozen` step installed from the STALE
   uv.lock — i.e. the UPSTREAM stack, not the forks (`--frozen` never
   compares the lock against pyproject).  STEP 40's bare `uv run` was
   simply the first command that resolved the new pyproject.  A build
   only bakes the fork stack after the relock below is committed.
2. **vLLM fork build failures** (CUDA arch flags, flashinfer version
   expectations): the fork is cut from an older vLLM than our 0.25.1 —
   downstream packages that import vllm internals (nemo_rl's
   vllm_backend extension!) may need version guards.
3. **`AutoModelForMultimodalLM`**: Apertus v1.5 loads via a NEW auto
   class.  NeMo-RL's dtensor/automodel worker load path must be checked
   (and possibly extended) to construct it; the VLM pipeline
   (`vlm_grpo-*-automodel` recipes) is the closest in-tree precedent.
   Text-only GRPO through the multimodal class is untested territory.
4. The model repo is **gated** — the HF token used by the driver must
   have accepted the license.
