# Operational Invariants (STRICT)

## Ponytail Mode (STRICT INVARIANT)
- **Active on every response**. Zero drift allowed.
- Lazy senior developer mode: YAGNI, standard library/native features first, minimal diffs, reuse existing functions/DB models before writing new abstractions.
- Never write new dispatchers, tasks, or classes when an existing allowlisted enqueue function or registered generic Celery operation solves the problem.
- Output code first. Explanation max 3 short lines. Never explain longer than code.

## Caveman Mode (STRICT INVARIANT)
- **Active on every response**. Zero drift allowed.
- Ultra-terse, token-minimal smart caveman style: drop articles (a/an/the), filler, pleasantries, preambles, and conversational narrative.
- Full technical substance, exact code, verbatim paths, commands, IDs, and errors.


# Python environment

This project uses a local virtual environment in `.venv`.

When running Python commands, use the project interpreter directly:

- Windows: `.venv\Scripts\python.exe`
- Linux/macOS: `.venv/bin/python`

Do not use bare `python` or system Python unless explicitly requested.
Install packages into `.venv`, not globally.

## Security & HPC Authentication (CRITICAL)
- **Zero Password Gathering**: Never ask the user for passwords, never write scripts or code that prompt for or collect passwords, and never store or handle credentials. Password gathering activity is a strictly bannable offense flagged by IT.
- **Use Established Repo Mechanisms**: All HPC communication (TRH408, Falcon, etc.) must use the repository's confirmed, non-interactive SSH tooling:
  - `core.openssh.OpenSSHClient` with `batch_mode=True` and existing key-based authentication (`~/.ssh` / ssh-agent / pre-authorized keys in `HPCSystem`).
  - Never disable batch mode to prompt for interactive authentication or passphrases.
  - Rely exclusively on established system profiles and database configs (`HPCSystem`).

## Deployment Guidelines
- Update deployed HPC checkouts via `gitupdate` using the configured non-interactive SSH profile in `HPCSystem`, `~/.ssh/config`, or gitignored `AGENTS.local.md` / `GEMINI.local.md`.

## HPC Inspection and Debugging (STRICT)
- Agents must never inspect HPC systems through direct SSH, scheduler commands, or `OpenSSHClient`.
- Use approved Celery operations for status checks, diagnostics, log retrieval, and artifact ingestion.
- Job inspection uses `enqueue_analysis_request("job_diagnostics", {"job_id": id, "tail_lines": 200})`; inspect returned local JSON artifact after completion.
- Registered remote artifacts use existing `enqueue_artifact_ingest`; do not invent download routes.
- Agents may read local database records and locally cached artifacts.
- Celery operations must accept database IDs and bounded options only. Never accept raw commands, arbitrary paths, shell syntax, or environment variables.
- Worker must derive HPC connection, scheduler ID, and paths from `HPCSystem`, `JobRecord`, `WorkflowRun`, and `Artifact`.
- Remote reads must be bounded, read-only, audited, and confined to recorded job directories. Limited worker-generated output listings are allowed; recursive browsing is not.
- Report Celery-queued, scheduler-verified, database-recorded, cached, and ingested states separately. Include status source and timestamp.
- If approved operation fails or is unavailable, report `not verified`. Never fall back to direct remote access.
- Never change authentication, `batch_mode`, or Celery worker state.

## POV-Ray report figures (persistent user preference)
- Manuscript and report structure figures must be rendered with POV-Ray. Never mix ASE/Matplotlib coordinate projections into a POV-Ray figure set or silently use them as a fallback.
- Use the repository's established non-interactive ASE scene export plus POV-Ray `/NR /RENDER <scene.ini> /EXIT` workflow; desktop or GUI control is not required.
- Preserve the source-coordinate, geometry-property, scene and rendered-image lineage. Treat a failed POV-Ray render as a blocker and verify every structure image before composing manuscript panels.

## Job submission pattern (persistent user preference)
- **The user owns the worker process**: the global Celery worker is always started, stopped, and restarted by the user in their terminal. Agents must never launch, terminate, replace, or otherwise manage a Celery worker process.
- **Worker Does NOT Need Restarting for New Jobs**: The running Celery worker is persistent. New calculation campaigns, parameter sets, and job submissions do NOT require restarting the worker. Never argue with the user or demand a worker restart to submit work.
- **Web Submit Pipeline & Submission Routines**: Submitting via the web submit pipeline (`JobSubmitForm` / `JobSubmitView` / `submit_slurm_workflow_run`) and the existing Celery dispatchers (`submit_analysis_request_task`) are standard routes that add jobs onto the active running infrastructure seamlessly without worker resets.
- **Celery is mandatory for async queue operations**: requested post-processing batches, Multiwfn analysis, IFCT workups, and background recoveries must publish to the Celery `submissions` queue. When using the interactive/web submit pipeline directly for workflow submissions, use the established form/backend submission routines.
- **Zero manual scripts & zero user proxying**: Never generate standalone `.sh` batch scripts, Python runners, or command-line instructions for the user to execute manually via `sbatch` or bash on HPC clusters. Asking or instructing the user to run a batch job or command directly on an HPC cluster is strictly treated as an out-of-band submission violation.
- **No out-of-band execution or shortcuts**: Never bypass Celery or write ad-hoc scripts under the justification of saving compute time, skipping redundant steps, or reusing wavefunctions. If a native recovery route does not exist, use the standard Celery operation (e.g. `full_ct_workup`).
- The permitted flow is: Codex/producer calls an `enqueue_*` function (prefer `jobs.tasks.enqueue_analysis_request`) -> Celery task is routed to `submissions` -> the submission worker checks capacity -> only that worker calls `jobs.submission.submit_slurm_workflow_run` and communicates with the HPC system.
- Calling an enqueue function from `manage.py shell` is allowed; calling the underlying submission helper from it is not. A dry-run manifest, locally constructed workflow record, or direct SSH command does not fulfil a submission request.
- Never work around a local worker authentication failure by submitting generated Slurm scripts manually. Leave the Celery tasks retrying and restore the worker's supported batch-mode key/agent access.
- Before publishing, verify that the worker is online and has registered the task. After publishing, record the Celery task ID and database marker, and verify that the message is routed to `submissions`. Do not describe work as queued unless this evidence exists.
- Use the stable generic dispatcher for new campaigns that fit an existing analysis operation. Do not add a campaign-specific Celery task merely to change source IDs or calculation parameters; this avoids worker restarts for ordinary submission requests.
- An SSH/authentication failure in the current agent process is not a reason to stop at a manifest: enqueue locally and report the worker's actual status. Keep all SSH non-interactive under the security rules above; never collect credentials.
- Mulliken-only recovery uses `ground_state_mulliken_only=True`, existing final-geometry wavefunctions, `standard` partition / `short` QoS, one task, 20 minutes, and the existing maximum of four active short-queue jobs. Preserve source-run and geometry lineage and skip existing recovery jobs.
- Report separately: queued locally, submitted to Slurm (with job IDs), completed, and ingested. Never call locally queued work submitted to Slurm or completed.
- Mulliken is the default population analysis for future ORCA runs. Keep Löwdin separately labelled; recovered Multiwfn Mulliken should populate the default database charge aliases and not be replaced by later Löwdin ingestion.
- Ground/excited atomic-charge comparisons must use Multiwfn Mulliken populations from the corresponding wavefunctions. Do not substitute ORCA population tables, NTO orbital percentages, or IFCT-derived estimates for these atomic charges.
