"""Experiment planning and analysis helpers for the PAPPO top-paper track."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from pappo.trajectory import AgentTrajectory, split_tool_call_turns
from pappo.trajectory import trajectory_from_mapping


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_MANIFEST = Path("data/realrepofix_100_manifest.jsonl")


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark/split target in the top-paper plan."""

    name: str
    manifest: str
    train_limit: int
    eval_limit: int
    family: str = "realrepofix"
    notes: str = ""


@dataclass(frozen=True)
class MethodSpec:
    """One method or baseline target in the top-paper plan."""

    name: str
    family: str
    runner: str | None
    notes: str = ""


@dataclass(frozen=True)
class TopPaperConfig:
    """Configuration for Qwen-only top-paper experiment planning."""

    models: tuple[str, ...]
    benchmarks: tuple[BenchmarkSpec, ...]
    methods: tuple[MethodSpec, ...]
    seeds: tuple[int, ...]
    output_root: str = "data/top_paper_qwen"
    backend: str = "hf"
    max_new_tokens: int = 384
    temperature: float = 0.7
    max_length: int = 768
    updates: int = 5
    num_rollouts_per_task: int = 2
    learning_rate: float = 1e-4
    ppo_epochs: int = 1
    local_prior: float = 0.25
    local_files_only: bool = True


@dataclass(frozen=True)
class ExperimentJob:
    """One planned experiment job."""

    job_id: str
    model: str
    benchmark: str
    method: str
    seed: int
    status: str
    output_dir: str
    command: tuple[str, ...] | None
    notes: str = ""


@dataclass(frozen=True)
class ExperimentPlan:
    """Expanded experiment plan plus command metadata."""

    models: list[str]
    benchmarks: list[BenchmarkSpec]
    methods: list[MethodSpec]
    seeds: list[int]
    jobs: list[ExperimentJob]
    setup_commands: list[tuple[str, ...]]
    benchmark_gaps: list[dict[str, str]]

    def to_dict(self) -> dict:
        runnable = [job for job in self.jobs if job.command]
        needs_runner = [job for job in self.jobs if job.status == "needs_runner"]
        return {
            "status": "top_paper_plan_created",
            "models": self.models,
            "benchmarks": [asdict(benchmark) for benchmark in self.benchmarks],
            "methods": [asdict(method) for method in self.methods],
            "seeds": self.seeds,
            "job_count": len(self.jobs),
            "runnable_job_count": len(runnable),
            "needs_runner_count": len(needs_runner),
            "setup_commands": [list(command) for command in self.setup_commands],
            "benchmark_gaps": self.benchmark_gaps,
            "jobs": [
                {**asdict(job), "command": list(job.command) if job.command else None}
                for job in self.jobs
            ],
        }


DEFAULT_TOP_PAPER_CONFIG = TopPaperConfig(
    models=(DEFAULT_MODEL,),
    benchmarks=(
        BenchmarkSpec(
            name="realrepofix_100",
            manifest=str(DEFAULT_MANIFEST),
            train_limit=70,
            eval_limit=30,
            notes="Current published split.",
        ),
        BenchmarkSpec(
            name="realrepofix_300",
            manifest="data/realrepofix_300_manifest.jsonl",
            train_limit=240,
            eval_limit=60,
            notes="Requires generating or importing RealRepoFix-300.",
        ),
        BenchmarkSpec(
            name="swe_bench_lite_subset",
            manifest="data/swe_bench_lite_subset_manifest.jsonl",
            train_limit=70,
            eval_limit=30,
            family="swe_bench_lite",
            notes="Requires SWE-bench adapter and task manifest.",
        ),
    ),
    methods=(
        MethodSpec(
            name="base_model",
            family="inference_baseline",
            runner="realrepo_lora_stage",
            notes="Base rollout/eval only.",
        ),
        MethodSpec(
            name="reward_weighted_lora",
            family="sft_baseline",
            runner="realrepo_lora_stage",
            notes="Existing weighted LoRA methods: trace/token/PAPPO variants.",
        ),
        MethodSpec(
            name="ppo_trace_reward",
            family="ppo_ablation",
            runner="pappo_ppo",
            notes="Use trace reward mode in PPO runner.",
        ),
        MethodSpec(
            name="ppo_token_broadcast_reward",
            family="ppo_ablation",
            runner="pappo_ppo",
            notes="Use token-broadcast reward mode in PPO runner.",
        ),
        MethodSpec(
            name="ppo_pappo_turn_local_reward",
            family="ppo_main",
            runner="pappo_ppo",
            notes="Current PAPPO true-PPO path.",
        ),
        MethodSpec(
            name="pappo_without_grouped_baseline",
            family="ablation",
            runner="pappo_ppo",
            notes="Use one rollout per task to select tool-mean critic.",
        ),
        MethodSpec(
            name="pappo_without_local_prior",
            family="ablation",
            runner="pappo_ppo",
            notes="Set local prior to 0.",
        ),
        MethodSpec(
            name="pappo_without_kl_ref_control",
            family="ablation",
            runner="pappo_ppo",
            notes="Set KL beta to 0.",
        ),
        MethodSpec(
            name="pappo_sft_weighted_lora",
            family="sft_baseline",
            runner="realrepo_lora_stage",
            notes="Use pappo_turn_v2 weighted LoRA.",
        ),
        MethodSpec(
            name="edit_only_vs_all_tool",
            family="ablation",
            runner="pappo_ppo",
            notes="Use all-tool action scope instead of edit-only.",
        ),
        MethodSpec(
            name="dpo_ipo_preference",
            family="preference_baseline",
            runner="preference_lora",
            notes="Build chosen/rejected pairs, then train a DPO/IPO-like preference LoRA.",
        ),
        MethodSpec(
            name="grpo_rloo_grouped_rollout",
            family="rl_baseline",
            runner="pappo_ppo",
            notes="Use leave-one-out grouped rollout critic mode.",
        ),
        MethodSpec(
            name="standard_ppo_terminal_reward",
            family="ppo_baseline",
            runner="pappo_ppo",
            notes="Use terminal reward mode in PPO runner.",
        ),
        MethodSpec(
            name="react_retry_test_heuristic",
            family="agent_baseline",
            runner="agent_pilot",
            notes="Use ReAct-style retry/test backend.",
        ),
        MethodSpec(
            name="reflexion_self_repair",
            family="agent_baseline",
            runner="agent_pilot",
            notes="Use Reflexion-style self-repair backend.",
        ),
    ),
    seeds=(0, 1, 2, 3, 4),
)


def _slug(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )


def _output_dir(config: TopPaperConfig, model: str, benchmark: BenchmarkSpec, method: MethodSpec, seed: int) -> str:
    return str(
        Path(config.output_root)
        / _slug(model)
        / benchmark.name
        / method.name
        / f"seed{seed}"
    )


def _ppo_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    method: MethodSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...]:
    num_rollouts = config.num_rollouts_per_task
    local_prior = config.local_prior
    kl_beta = 0.01
    critic_mode = "auto"
    if method.name == "pappo_without_grouped_baseline":
        num_rollouts = 1
    if method.name == "pappo_without_local_prior":
        local_prior = 0.0
    if method.name == "pappo_without_kl_ref_control":
        kl_beta = 0.0
    if method.name == "grpo_rloo_grouped_rollout":
        critic_mode = "rloo"
    reward_mode = "pappo_turn_local"
    action_scope = "edit"
    if method.name == "ppo_trace_reward":
        reward_mode = "trace"
    if method.name == "ppo_token_broadcast_reward":
        reward_mode = "token_broadcast"
    if method.name == "standard_ppo_terminal_reward":
        reward_mode = "terminal"
    if method.name == "edit_only_vs_all_tool":
        action_scope = "all_tools"

    command = [
        ".venv/bin/python",
        "scripts/run_realrepo_pappo_ppo.py",
        "--model",
        model,
        "--backend",
        config.backend,
        "--manifest",
        benchmark.manifest,
        "--train-limit",
        str(benchmark.train_limit),
        "--eval-limit",
        str(benchmark.eval_limit),
        "--updates",
        str(config.updates),
        "--num-rollouts-per-task",
        str(num_rollouts),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--max-length",
        str(config.max_length),
        "--learning-rate",
        str(config.learning_rate),
        "--ppo-epochs",
        str(config.ppo_epochs),
        "--local-prior",
        str(local_prior),
        "--kl-beta",
        str(kl_beta),
        "--critic-mode",
        critic_mode,
        "--reward-mode",
        reward_mode,
        "--action-scope",
        action_scope,
        "--no-normalize-advantages",
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
    ]
    if config.local_files_only:
        command.append("--local-files-only")
    return tuple(command)


def _lora_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    method: MethodSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...]:
    methods = ["trace", "token_broadcast", "pappo_turn_v2"]
    if method.name == "pappo_sft_weighted_lora":
        methods = ["pappo_turn_v2"]
    command = [
        ".venv/bin/python",
        "scripts/run_realrepo_lora_comparison.py",
        "--model",
        model,
        "--backend",
        config.backend,
        "--manifest",
        benchmark.manifest,
        "--train-limit",
        str(benchmark.train_limit),
        "--eval-limit",
        str(benchmark.eval_limit),
        "--seed",
        str(seed),
        "--methods",
        *methods,
        "--eval-base",
        "--num-rollouts-per-task",
        str(config.num_rollouts_per_task),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--max-length",
        str(config.max_length),
        "--learning-rate",
        str(config.learning_rate),
        "--output-dir",
        output_dir,
    ]
    if config.local_files_only:
        command.append("--local-files-only")
    return tuple(command)


def _agent_baseline_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    method: MethodSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...]:
    backend = "react_retry" if method.name == "react_retry_test_heuristic" else "reflexion"
    return (
        ".venv/bin/python",
        "scripts/run_llm_agent_pilot.py",
        "--model-path",
        model,
        "--backend",
        backend,
        "--manifest",
        benchmark.manifest,
        "--limit",
        str(benchmark.eval_limit),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--no-download",
        "--trajectories",
        str(Path(output_dir) / "eval_rollouts.jsonl"),
    )


def _base_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...]:
    command = [
        ".venv/bin/python",
        "scripts/run_realrepo_lora_comparison.py",
        "--model",
        model,
        "--backend",
        config.backend,
        "--manifest",
        benchmark.manifest,
        "--train-limit",
        str(benchmark.train_limit),
        "--eval-limit",
        str(benchmark.eval_limit),
        "--seed",
        str(seed),
        "--stage",
        "eval-base",
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--output-dir",
        output_dir,
    ]
    if config.local_files_only:
        command.append("--local-files-only")
    return tuple(command)


def _preference_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...]:
    pairs_path = Path("data/top_paper_qwen/setup") / benchmark.name / "preference_pairs.jsonl"
    command = [
        ".venv/bin/python",
        "scripts/run_preference_lora_baseline.py",
        "--model",
        model,
        "--pairs",
        str(pairs_path),
        "--output-dir",
        output_dir,
        "--max-length",
        str(config.max_length),
        "--learning-rate",
        str(config.learning_rate),
        "--seed",
        str(seed),
    ]
    if config.local_files_only:
        command.append("--local-files-only")
    return tuple(command)


def _swe_bench_lite_command(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    method: MethodSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...] | None:
    backends = {
        "base_model": "hf",
        "react_retry_test_heuristic": "react_retry",
        "reflexion_self_repair": "reflexion",
    }
    backend = backends.get(method.name)
    if backend is not None:
        return (
            ".venv/bin/python",
            "scripts/run_swe_bench_lite_pilot.py",
            "--model-path",
            model,
            "--backend",
            backend,
            "--manifest",
            benchmark.manifest,
            "--limit",
            str(benchmark.eval_limit),
            "--work-root",
            str(Path(output_dir) / "work"),
            "--trajectories",
            str(Path(output_dir) / "eval_rollouts.jsonl"),
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--temperature",
            str(config.temperature),
            "--timeout-seconds",
            "30",
            "--no-download",
        )
    if method.name in {"reward_weighted_lora", "pappo_sft_weighted_lora"}:
        lora_method = "pappo_turn_v2"
        return (
            ".venv/bin/python",
            "scripts/run_swe_bench_lora_baseline.py",
            "--model",
            model,
            "--manifest",
            benchmark.manifest,
            "--train-limit",
            str(benchmark.train_limit),
            "--eval-limit",
            str(benchmark.eval_limit),
            "--method",
            lora_method,
            "--output-dir",
            output_dir,
            "--max-length",
            str(config.max_length),
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--temperature",
            str(config.temperature),
            "--learning-rate",
            str(config.learning_rate),
            "--local-files-only",
        )
    if method.name == "dpo_ipo_preference":
        return (
            ".venv/bin/python",
            "scripts/run_swe_bench_preference_lora.py",
            "--model",
            model,
            "--manifest",
            benchmark.manifest,
            "--train-limit",
            str(benchmark.train_limit),
            "--output-dir",
            output_dir,
            "--max-length",
            str(config.max_length),
            "--learning-rate",
            str(config.learning_rate),
            "--local-files-only",
        )
    ppo_methods = {
        "ppo_trace_reward",
        "ppo_token_broadcast_reward",
        "ppo_pappo_turn_local_reward",
        "pappo_without_grouped_baseline",
        "pappo_without_local_prior",
        "pappo_without_kl_ref_control",
        "edit_only_vs_all_tool",
        "grpo_rloo_grouped_rollout",
        "standard_ppo_terminal_reward",
    }
    if method.name in ppo_methods:
        num_rollouts = config.num_rollouts_per_task
        local_prior = config.local_prior
        kl_beta = 0.01
        critic_mode = "auto"
        reward_mode = "pappo_turn_local"
        action_scope = "edit"
        if method.name == "pappo_without_grouped_baseline":
            num_rollouts = 1
        if method.name == "pappo_without_local_prior":
            local_prior = 0.0
        if method.name == "pappo_without_kl_ref_control":
            kl_beta = 0.0
        if method.name == "grpo_rloo_grouped_rollout":
            critic_mode = "rloo"
        if method.name == "ppo_trace_reward":
            reward_mode = "trace"
        if method.name == "ppo_token_broadcast_reward":
            reward_mode = "token_broadcast"
        if method.name == "standard_ppo_terminal_reward":
            reward_mode = "terminal"
        if method.name == "edit_only_vs_all_tool":
            action_scope = "all_tools"
        return (
            ".venv/bin/python",
            "scripts/run_swe_bench_pappo_ppo.py",
            "--model",
            model,
            "--manifest",
            benchmark.manifest,
            "--train-limit",
            str(benchmark.train_limit),
            "--eval-limit",
            str(benchmark.eval_limit),
            "--updates",
            str(config.updates),
            "--num-rollouts-per-task",
            str(num_rollouts),
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--temperature",
            str(config.temperature),
            "--max-length",
            str(config.max_length),
            "--learning-rate",
            str(config.learning_rate),
            "--ppo-epochs",
            str(config.ppo_epochs),
            "--local-prior",
            str(local_prior),
            "--kl-beta",
            str(kl_beta),
            "--critic-mode",
            critic_mode,
            "--reward-mode",
            reward_mode,
            "--action-scope",
            action_scope,
            "--no-normalize-advantages",
            "--seed",
            str(seed),
            "--output-dir",
            output_dir,
            "--local-files-only",
        )
    return None


def _command_for(
    config: TopPaperConfig,
    *,
    model: str,
    benchmark: BenchmarkSpec,
    method: MethodSpec,
    seed: int,
    output_dir: str,
) -> tuple[str, ...] | None:
    if benchmark.family == "swe_bench_lite":
        return _swe_bench_lite_command(
            config,
            model=model,
            benchmark=benchmark,
            method=method,
            seed=seed,
            output_dir=output_dir,
        )
    if benchmark.family != "realrepofix":
        return None
    if method.name == "base_model":
        return _base_command(
            config,
            model=model,
            benchmark=benchmark,
            seed=seed,
            output_dir=output_dir,
        )
    if method.runner == "realrepo_lora_stage":
        return _lora_command(
            config,
            model=model,
            benchmark=benchmark,
            method=method,
            seed=seed,
            output_dir=output_dir,
        )
    if method.runner == "pappo_ppo":
        return _ppo_command(
            config,
            model=model,
            benchmark=benchmark,
            method=method,
            seed=seed,
            output_dir=output_dir,
        )
    if method.runner == "agent_pilot":
        return _agent_baseline_command(
            config,
            model=model,
            benchmark=benchmark,
            method=method,
            seed=seed,
            output_dir=output_dir,
        )
    if method.runner == "preference_lora":
        return _preference_command(
            config,
            model=model,
            benchmark=benchmark,
            seed=seed,
            output_dir=output_dir,
        )
    return None


def _setup_commands(config: TopPaperConfig) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    for benchmark in config.benchmarks:
        if benchmark.name == "realrepofix_300":
            commands.append(
                (
                    ".venv/bin/python",
                    "scripts/generate_realrepofix.py",
                    "--root",
                    "data/realrepofix_300",
                    "--num-tasks",
                    "300",
                    "--seed",
                    "0",
                    "--manifest",
                    benchmark.manifest,
                    "--trajectories",
                    "data/realrepofix_300_trajectories.jsonl",
                    "--mixed-trajectories",
                )
            )
        if benchmark.name == "swe_bench_lite_subset":
            commands.append(
                (
                    ".venv/bin/python",
                    "scripts/prepare_swe_bench_lite_subset.py",
                    "--output",
                    benchmark.manifest,
                    "--limit",
                    str(benchmark.train_limit + benchmark.eval_limit),
                )
            )
        if benchmark.family == "realrepofix":
            trajectories = (
                "data/realrepofix_300_trajectories.jsonl"
                if benchmark.name == "realrepofix_300"
                else "data/realrepofix_100_mixed_trajectories.jsonl"
            )
            pairs = str(Path("data/top_paper_qwen/setup") / benchmark.name / "preference_pairs.jsonl")
            commands.append(
                (
                    ".venv/bin/python",
                    "scripts/build_preference_pairs.py",
                    "--trajectories",
                    trajectories,
                    "--output",
                    pairs,
                )
            )
    return commands


def _benchmark_gaps(config: TopPaperConfig) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for benchmark in config.benchmarks:
        if benchmark.family == "swe_bench_lite":
            gaps.append(
                {
                    "benchmark": benchmark.name,
                    "status": "needs_external_swe_data",
                    "needed": "Populate the SWE-bench Lite subset manifest with local repo_dir checkouts, source_file paths, candidate/gold patches, and test commands before running jobs.",
                }
            )
    return gaps


def build_experiment_plan(config: TopPaperConfig = DEFAULT_TOP_PAPER_CONFIG) -> ExperimentPlan:
    """Expand a top-paper config into runnable and not-yet-runnable jobs."""

    jobs: list[ExperimentJob] = []
    for model in config.models:
        for benchmark in config.benchmarks:
            for method in config.methods:
                for seed in config.seeds:
                    output_dir = _output_dir(config, model, benchmark, method, seed)
                    command = _command_for(
                        config,
                        model=model,
                        benchmark=benchmark,
                        method=method,
                        seed=seed,
                        output_dir=output_dir,
                    )
                    status = "planned" if command else "needs_runner"
                    notes = method.notes
                    if benchmark.family != "realrepofix":
                        notes = f"{benchmark.notes} {notes}".strip()
                    jobs.append(
                        ExperimentJob(
                            job_id="__".join(
                                [
                                    _slug(model),
                                    benchmark.name,
                                    method.name,
                                    f"seed{seed}",
                                ]
                            ),
                            model=model,
                            benchmark=benchmark.name,
                            method=method.name,
                            seed=seed,
                            status=status,
                            output_dir=output_dir,
                            command=command,
                            notes=notes,
                        )
                    )
    return ExperimentPlan(
        models=list(config.models),
        benchmarks=list(config.benchmarks),
        methods=list(config.methods),
        seeds=list(config.seeds),
        jobs=jobs,
        setup_commands=_setup_commands(config),
        benchmark_gaps=_benchmark_gaps(config),
    )


def bootstrap_success_ci(
    rewards: list[float],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return a deterministic bootstrap CI for mean success."""

    if not rewards:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    samples = []
    n = len(rewards)
    for _ in range(iterations):
        samples.append(mean(rng.choice(rewards) for _item in range(n)))
    samples.sort()
    low_index = max(0, int((alpha / 2.0) * iterations))
    high_index = min(iterations - 1, int((1.0 - alpha / 2.0) * iterations))
    return {
        "mean": float(mean(rewards)),
        "low": float(samples[low_index]),
        "high": float(samples[high_index]),
    }


def paired_win_loss(
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    """Compute paired task-level win/loss/tie counts."""

    common = sorted(set(baseline) & set(candidate))
    wins = 0
    losses = 0
    ties = 0
    for key in common:
        if candidate[key] > baseline[key]:
            wins += 1
        elif candidate[key] < baseline[key]:
            losses += 1
        else:
            ties += 1
    pairs = len(common)
    return {
        "pairs": pairs,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / pairs) if pairs else 0.0,
        "loss_rate": float(losses / pairs) if pairs else 0.0,
    }


def sign_test_p_value(*, wins: int, losses: int) -> float:
    """Two-sided exact sign-test p-value with ties removed."""

    trials = int(wins) + int(losses)
    if trials <= 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return float(min(1.0, 2.0 * probability))


def load_trajectory_jsonl(path: Path) -> list[AgentTrajectory]:
    """Load trajectory JSONL for top-paper analysis."""

    trajectories: list[AgentTrajectory] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                trajectories.append(trajectory_from_mapping(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"failed to parse {path}:{line_number}") from exc
    return trajectories


def task_rewards(trajectories: list[AgentTrajectory]) -> dict[str, float]:
    """Map task ids to final rewards for paired analysis."""

    rewards: dict[str, float] = {}
    for trajectory in trajectories:
        task_id = str(
            trajectory.metadata.get("task_id")
            or trajectory.metadata.get("repo_dir")
            or trajectory.trajectory_id
        )
        rewards[task_id] = float(trajectory.final_reward)
    return rewards


def _task_id(trajectory: AgentTrajectory) -> str:
    return str(
        trajectory.metadata.get("task_id")
        or trajectory.metadata.get("repo_dir")
        or trajectory.trajectory_id
    )


def _turn_patch_applied(turn) -> bool:
    return bool(dict(turn.metadata.get("result_metadata", {})).get("patch_applied", False))


def _turn_pytest_reward(turn) -> float | None:
    metadata = dict(turn.metadata.get("result_metadata", {}))
    if "pytest_reward" not in metadata:
        return None
    return float(metadata["pytest_reward"])


def failure_categories(trajectories: list[AgentTrajectory]) -> dict[str, int]:
    """Count coarse failure categories for unsuccessful trajectories."""

    categories = {
        "failed_patch": 0,
        "no_patch": 0,
        "test_failed_after_patch": 0,
    }
    for trajectory in trajectories:
        if float(trajectory.final_reward) > 0.0:
            continue
        turns = split_tool_call_turns(trajectory)
        edit_turns = [turn for turn in turns if turn.tool_name == "edit"]
        applied_edits = [turn for turn in edit_turns if _turn_patch_applied(turn)]
        if applied_edits:
            categories["failed_patch"] += 1
        else:
            categories["no_patch"] += 1
        if applied_edits and any(
            turn.tool_name == "run_test" and (_turn_pytest_reward(turn) or 0.0) <= 0.0
            for turn in turns
        ):
            categories["test_failed_after_patch"] += 1
    return categories


def _case_summary(trajectory: AgentTrajectory) -> dict[str, object]:
    turns = split_tool_call_turns(trajectory)
    return {
        "task_id": _task_id(trajectory),
        "trajectory_id": trajectory.trajectory_id,
        "reward": float(trajectory.final_reward),
        "tool_pattern": ">".join(turn.tool_name for turn in turns),
        "edit_count": sum(1 for turn in turns if turn.tool_name == "edit"),
        "test_count": sum(1 for turn in turns if turn.tool_name == "run_test"),
        "patch_applied": any(
            turn.tool_name == "edit" and _turn_patch_applied(turn)
            for turn in turns
        ),
    }


def failure_case_examples(
    *,
    baseline: list[AgentTrajectory],
    candidate: list[AgentTrajectory],
    limit: int = 5,
) -> dict[str, list[dict[str, object]]]:
    """Return before/after examples for improved, regressed, and persistent failures."""

    baseline_by_task = {_task_id(trajectory): trajectory for trajectory in baseline}
    candidate_by_task = {_task_id(trajectory): trajectory for trajectory in candidate}
    improved: list[dict[str, object]] = []
    regressed: list[dict[str, object]] = []
    persistent_failures: list[dict[str, object]] = []
    for task_id in sorted(set(baseline_by_task) & set(candidate_by_task)):
        base = baseline_by_task[task_id]
        cand = candidate_by_task[task_id]
        base_reward = float(base.final_reward)
        cand_reward = float(cand.final_reward)
        row = {
            "task_id": task_id,
            "baseline_reward": base_reward,
            "candidate_reward": cand_reward,
            "baseline": _case_summary(base),
            "candidate": _case_summary(cand),
        }
        if cand_reward > base_reward:
            improved.append(row)
        elif cand_reward < base_reward:
            regressed.append(row)
        elif cand_reward <= 0.0:
            persistent_failures.append(row)
    return {
        "improved": improved[:limit],
        "regressed": regressed[:limit],
        "persistent_failures": persistent_failures[:limit],
    }


def mechanism_metrics(trajectories: list[AgentTrajectory]) -> dict[str, float]:
    """Compute mechanism-analysis metrics over agent trajectories."""

    edit_successes: list[float] = []
    patch_applied_values: list[float] = []
    failed_patches: list[float] = []
    tests_after_edit: list[float] = []
    retries: list[float] = []
    test_counts: list[float] = []
    patch_tokens: list[float] = []
    pattern_counts: dict[str, int] = {}

    for trajectory in trajectories:
        turns = split_tool_call_turns(trajectory)
        tool_names = [turn.tool_name for turn in turns]
        edit_count = sum(1 for tool_name in tool_names if tool_name == "edit")
        test_count = sum(1 for tool_name in tool_names if tool_name == "run_test")
        retries.append(float(max(0, edit_count - 1)))
        test_counts.append(float(test_count))
        if tool_names:
            key = "tool_pattern:" + ">".join(tool_names[:3])
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        for index, turn in enumerate(turns):
            if turn.tool_name != "edit":
                continue
            result_metadata = dict(turn.metadata.get("result_metadata", {}))
            call_metadata = dict(turn.metadata.get("call_metadata", {}))
            patch_applied = bool(result_metadata.get("patch_applied", False))
            patch_applied_values.append(float(patch_applied))
            edit_successes.append(float(patch_applied and trajectory.final_reward > 0.0))
            failed_patches.append(float(patch_applied and trajectory.final_reward <= 0.0))
            if "generated_tokens" in call_metadata:
                patch_tokens.append(float(call_metadata["generated_tokens"]))
            elif turn.tool_result:
                patch_tokens.append(float(len(turn.tool_result.split())))
            later_tests = [later for later in turns[index + 1 :] if later.tool_name == "run_test"]
            if later_tests:
                passed = any(
                    float(
                        dict(later.metadata.get("result_metadata", {})).get(
                            "pytest_reward",
                            trajectory.final_reward,
                        )
                    )
                    > 0.0
                    for later in later_tests
                )
                tests_after_edit.append(float(passed))

    metrics = {
        "trajectory_count": float(len(trajectories)),
        "edit_success_rate": float(mean(edit_successes)) if edit_successes else 0.0,
        "patch_apply_rate": float(mean(patch_applied_values)) if patch_applied_values else 0.0,
        "failed_patch_rate": float(mean(failed_patches)) if failed_patches else 0.0,
        "test_pass_after_edit_rate": float(mean(tests_after_edit)) if tests_after_edit else 0.0,
        "avg_retries": float(mean(retries)) if retries else 0.0,
        "avg_test_calls": float(mean(test_counts)) if test_counts else 0.0,
        "avg_patch_tokens": float(mean(patch_tokens)) if patch_tokens else 0.0,
    }
    metrics.update({key: float(value) for key, value in sorted(pattern_counts.items())})
    return metrics


def write_plan(path: Path, plan: ExperimentPlan) -> None:
    """Write a JSON experiment plan."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
