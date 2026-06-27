from __future__ import annotations

from pathlib import Path

from pappo.turn_critic import GroupMeanTurnCritic, MeanTurnCritic


def test_mean_turn_critic_predicts_tool_specific_means(tmp_path: Path) -> None:
    critic = MeanTurnCritic()
    critic.fit(
        tool_names=["edit", "edit", "run_test"],
        targets=[1.0, 0.0, 1.0],
    )

    assert critic.predict("edit") == 0.5
    assert critic.predict("run_test") == 1.0
    assert critic.predict("search") == 0.0

    path = tmp_path / "critic.json"
    critic.save(path)
    loaded = MeanTurnCritic.load(path)
    assert loaded.predict("edit") == 0.5


def test_group_mean_turn_critic_prefers_task_local_means(tmp_path: Path) -> None:
    critic = GroupMeanTurnCritic()
    critic.fit(
        group_keys=["task-a", "task-a", "task-b"],
        tool_names=["edit", "edit", "edit"],
        targets=[1.0, -1.0, 1.0],
    )

    assert critic.predict("task-a", "edit") == 0.0
    assert critic.predict("task-b", "edit") == 1.0
    assert critic.predict("unseen-task", "edit") == 1.0 / 3.0

    path = tmp_path / "group_critic.json"
    critic.save(path)
    loaded = GroupMeanTurnCritic.load(path)
    assert loaded.predict("task-a", "edit") == 0.0
