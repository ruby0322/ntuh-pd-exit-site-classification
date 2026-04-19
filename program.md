# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — shared constants (`TIME_BUDGET`, `IMAGE_SIZE`, `BINARY_INFECTION_CLASS`) and dataset validation. Run once to confirm data is healthy; do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `summarize_results.py` — derives the current screening frontier and next-step hints from `results.tsv`.
4. **Verify data exists**: Run `python prepare.py` to confirm `./dataset/` has the expected 5-class ImageFolder layout and all images are readable.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py` (venv activated).

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness (`evaluate()`), shared constants, and dataset utilities.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The primary goal is to maximise `bin_acc`** (binary infection screening accuracy). This is the clinically important metric: correctly distinguishing infection-positive (class 4) from infection-negative (classes 0–3).

**Secondary goal: also improve `mc_acc`** (5-class multiclass accuracy) without sacrificing `bin_acc`. A change that improves `bin_acc` is always worth keeping even if `mc_acc` is flat. A change that only improves `mc_acc` is worth keeping if `bin_acc` does not drop.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful accuracy gains, but it should not blow up dramatically (current runs use ≤ 1 GB).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh complexity cost against metric gains. A 0.001 bin_acc gain from 20 extra lines of hacky code? Probably not worth it. A 0.001 bin_acc gain from deleting code? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

`train.py` prints **all batch/epoch logs to stdout** (use `flush=True` so lines appear immediately). Do not redirect stdout unless you explicitly want a file copy; the human should see training live.

By default, **wall-clock training is capped at 300 seconds** (`--max-train-seconds 300`). Training stops mid-epoch if needed; evaluation still runs afterward (a few extra seconds).

Summary footer:

```
---
mc_acc:               0.563600
bin_acc:              0.712300
train_seconds:        300.2
train_stopped_budget: true
peak_vram_mb:         1234.5
arch:                 baseline
optimizer:            sgd
```

Extract metrics (from the terminal scrollback or a log you chose to save):

```
grep "^mc_acc:\|^bin_acc:\|^peak_vram_mb:\|^train_stopped_budget:" 
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	mc_acc	bin_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `mc_acc` — 5-class multiclass accuracy, e.g. `0.568000`; use `0.000000` for crashes
3. `bin_acc` — binary infection accuracy, e.g. `0.896000`; use `0.000000` for crashes
4. peak VRAM in GB, round to 1 decimal (divide `peak_vram_mb` by 1024); use `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

**Keep/discard rules:**
- **keep** if `bin_acc` is strictly higher than the current best, regardless of `mc_acc`
- **keep** if `bin_acc` is equal to the current best AND `mc_acc` improves (tie-break)
- **keep** if `bin_acc` is equal AND `mc_acc` is equal AND the change simplifies the code
- **discard** otherwise (flat or regression on both metrics)
- **crash** if the run throws an exception or produces no footer output

Example:

```
commit	mc_acc	bin_acc	memory_gb	status	description
a1b2c3d	0.286000	0.811000	0.3	keep	baseline (wide arch, SGD lr=0.01)
b2c3d4e	0.568000	0.896000	0.5	keep	wider channels — bin_acc +8.5pp
c3d4e5f	0.580000	0.822000	0.2	discard	lite arch — bin_acc regression vs best
d4e5f6g	0.000000	0.000000	0.0	crash	doubled width (OOM)
```

## Loop summary artifact

After every update to `results.tsv`, regenerate the derived analysis files:

```bash
python summarize_results.py
```

This writes:

- `analysis_summary.json` — machine-readable state for the agent
- `analysis_summary.md` — short human-readable brief

The JSON is the one you should read before choosing the next experiment. Use it to:

- identify the current screening frontier
- spot promising near-miss runs with high `mc_acc`
- avoid repeating losing regions of the search space without a new independent idea
- prefer single-axis changes around the current best recipe

Do not commit `results.tsv`, `analysis_summary.json`, or `analysis_summary.md`.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Read `analysis_summary.json` if it exists. Use it to choose the next experiment from the current frontier, promising near misses, or one new orthogonal idea.
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. git commit
5. Run the experiment in the **foreground** so logs stay visible: `python3 train.py` (venv activated). Optional: `python3 train.py 2>&1 | tee run.log` if you want a file **and** a live stream — do **not** use `> run.log` alone, which hides output.
6. Read out the results from the footer (or from `run.log` if you used `tee`).
7. If the run crashed, read the traceback from the terminal (or `tail -n 80 run.log` if you captured output). If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. Regenerate the derived summaries with `python summarize_results.py`
10. Apply the **keep/discard rules** from the "Logging results" section:
   - **keep** → the branch advances; keep the git commit and continue from here
   - **discard** → `git reset --hard HEAD~1` to revert to the previous commit, then try a different idea
   - **crash** → fix if trivial, otherwise discard and revert
11. The new "current best" after a keep is the bin_acc (and mc_acc tie-break) of that commit, and `analysis_summary.json` becomes the starting point for the next loop iteration

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Training is capped at **5 minutes wall-clock by default** inside `train.py`. If you disabled the cap (`--max-train-seconds 0`) and a run drags on far beyond ~10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
