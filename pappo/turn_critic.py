"""Turn-level value estimators for PAPPO-PPO."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


class MeanTurnCritic:
    """A tiny tool-conditioned scalar critic for PPO smoke tests."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.values = values or {}

    def fit(self, *, tool_names: list[str], targets: list[float]) -> None:
        grouped: dict[str, list[float]] = {}
        for tool_name, target in zip(tool_names, targets, strict=True):
            grouped.setdefault(tool_name, []).append(float(target))
        self.values = {
            tool_name: float(mean(tool_targets))
            for tool_name, tool_targets in grouped.items()
        }

    def predict(self, tool_name: str) -> float:
        return float(self.values.get(tool_name, 0.0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.values, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "MeanTurnCritic":
        return cls(json.loads(path.read_text(encoding="utf-8")))


class GroupMeanTurnCritic:
    """Task-local scalar critic with tool-level fallback."""

    def __init__(
        self,
        group_values: dict[str, float] | None = None,
        fallback_values: dict[str, float] | None = None,
    ) -> None:
        self.group_values = group_values or {}
        self.fallback_values = fallback_values or {}

    @staticmethod
    def _key(group_key: str, tool_name: str) -> str:
        return json.dumps([str(group_key), str(tool_name)], separators=(",", ":"))

    def fit(
        self,
        *,
        group_keys: list[str],
        tool_names: list[str],
        targets: list[float],
    ) -> None:
        grouped: dict[str, list[float]] = {}
        fallback: dict[str, list[float]] = {}
        for group_key, tool_name, target in zip(
            group_keys,
            tool_names,
            targets,
            strict=True,
        ):
            grouped.setdefault(self._key(group_key, tool_name), []).append(float(target))
            fallback.setdefault(tool_name, []).append(float(target))
        self.group_values = {
            key: float(mean(group_targets))
            for key, group_targets in grouped.items()
        }
        self.fallback_values = {
            tool_name: float(mean(tool_targets))
            for tool_name, tool_targets in fallback.items()
        }

    def predict(self, group_key: str, tool_name: str) -> float:
        key = self._key(group_key, tool_name)
        if key in self.group_values:
            return float(self.group_values[key])
        return float(self.fallback_values.get(tool_name, 0.0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "group_values": self.group_values,
                    "fallback_values": self.fallback_values,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "GroupMeanTurnCritic":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            group_values=dict(payload.get("group_values", {})),
            fallback_values=dict(payload.get("fallback_values", {})),
        )


class RLOOTurnCritic:
    """Leave-one-out grouped scalar critic with tool-level fallback."""

    def __init__(
        self,
        sample_values: dict[str, float] | None = None,
        group_values: dict[str, float] | None = None,
        fallback_values: dict[str, float] | None = None,
    ) -> None:
        self.sample_values = sample_values or {}
        self.group_values = group_values or {}
        self.fallback_values = fallback_values or {}

    @staticmethod
    def _group_key(group_key: str, tool_name: str) -> str:
        return json.dumps([str(group_key), str(tool_name)], separators=(",", ":"))

    def fit(
        self,
        *,
        sample_ids: list[str],
        group_keys: list[str],
        tool_names: list[str],
        targets: list[float],
    ) -> None:
        grouped: dict[str, list[tuple[str, float]]] = {}
        fallback: dict[str, list[float]] = {}
        for sample_id, group_key, tool_name, target in zip(
            sample_ids,
            group_keys,
            tool_names,
            targets,
            strict=True,
        ):
            grouped.setdefault(
                self._group_key(group_key, tool_name),
                [],
            ).append((str(sample_id), float(target)))
            fallback.setdefault(tool_name, []).append(float(target))

        self.fallback_values = {
            tool_name: float(mean(tool_targets))
            for tool_name, tool_targets in fallback.items()
        }
        self.group_values = {
            key: float(mean(target for _sample_id, target in group_targets))
            for key, group_targets in grouped.items()
        }
        sample_values: dict[str, float] = {}
        for key, group_targets in grouped.items():
            for sample_id, target in group_targets:
                others = [
                    other_target
                    for other_id, other_target in group_targets
                    if other_id != sample_id
                ]
                if others:
                    sample_values[sample_id] = float(mean(others))
                else:
                    tool_name = str(json.loads(key)[1])
                    sample_values[sample_id] = self.fallback_values.get(
                        tool_name,
                        self.group_values[key],
                    )
        self.sample_values = sample_values

    def predict(self, sample_id: str, group_key: str, tool_name: str) -> float:
        if sample_id in self.sample_values:
            return float(self.sample_values[sample_id])
        key = self._group_key(group_key, tool_name)
        if key in self.group_values:
            return float(self.group_values[key])
        return float(self.fallback_values.get(tool_name, 0.0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sample_values": self.sample_values,
                    "group_values": self.group_values,
                    "fallback_values": self.fallback_values,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "RLOOTurnCritic":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            sample_values=dict(payload.get("sample_values", {})),
            group_values=dict(payload.get("group_values", {})),
            fallback_values=dict(payload.get("fallback_values", {})),
        )
