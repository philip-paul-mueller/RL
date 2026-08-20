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

## Relock procedure (the image build does it — RELOCK build-arg)

`uv lock` needs uv 0.11.28 + torch/toolchain for the vLLM fork's
metadata build, and no local image exists to run it in — so the build
container itself does the relock (Containerfile arg added in
alps-extended-images `a83467e`):

```bash
# 1. one-time relock build (make sure the clone layer is NOT served from
#    cache — the branch HEAD moved; --no-cache is the blunt safe option)
podman build ... --build-arg NEMORL_COMMIT=apertus-stack --build-arg RELOCK=1 ...
# expect a LONG build: the swiss vllm compiles its CUDA kernels from source

# 2. extract the fresh lock from the image and commit it — MANDATORY:
#    the runtime driver checks the branch out in-container; a committed
#    lock that differs from the baked venvs re-resolves into the enroot
#    overlay (or fails offline)
podman create --name relock <image>
podman cp relock:/workdir/nemo_rl/uv.lock uv.lock && podman rm relock
git add uv.lock && git commit -s -m "chore(apertus): relock with swiss-ai forks"
git push fork apertus-stack

# 3. (optional but recommended) rebuild once WITHOUT RELOCK to confirm
#    the committed lock reproduces the image as a pure consumer
```

If `uv lock` fails in step 1 with another version conflict, it is the
sglang shape again (risk #1 below) — fix with another
override-dependencies entry and rebuild.

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
