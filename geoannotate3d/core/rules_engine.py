"""core/rules_engine.py — Clasificación automática por reglas geoespaciales."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Rule:
    id:           int   = 0
    name:         str   = ""
    target_class: int   = 1
    use_agl:      bool  = False
    agl_min:      float = 0.0
    agl_max:      float = 100.0
    use_z:        bool  = False
    z_min:        float = -9999.0
    z_max:        float =  9999.0
    use_intensity: bool  = False
    int_min:       float = 0.0
    int_max:       float = 1.0
    use_return:    bool  = False
    return_first_only: bool = True
    only_unlabeled: bool = True
    enabled:        bool = True


class RulesEngine:
    def __init__(self):
        self._rules: List[Rule] = []; self._next_id = 1

    def add_rule(self, rule: Rule) -> None:
        rule.id = self._next_id; self._next_id += 1; self._rules.append(rule)

    def remove_rule(self, rule_id: int) -> None:
        self._rules = [r for r in self._rules if r.id != rule_id]

    def clear(self) -> None: self._rules.clear()

    @property
    def rules(self): return self._rules

    def apply(self, pc, labels: np.ndarray, terrain_model=None) -> int:
        xyz = pc.xyz
        if xyz is None or len(xyz) == 0: return 0
        agl_vals = None
        if (terrain_model is not None and terrain_model.ready and
                any(r.use_agl and r.enabled for r in self._rules)):
            agl_vals = terrain_model.agl(xyz)
        total = 0
        for rule in self._rules:
            if not rule.enabled: continue
            mask = self._mask(rule, xyz, labels, pc, agl_vals)
            n = int(mask.sum())
            if n > 0: labels[mask] = rule.target_class; total += n
        return total

    def preview(self, pc, labels, rule, terrain_model=None):
        xyz = pc.xyz
        if xyz is None: return np.zeros(0, bool)
        agl_vals = terrain_model.agl(xyz) if (rule.use_agl and terrain_model and terrain_model.ready) else None
        return self._mask(rule, xyz, labels, pc, agl_vals)

    def _mask(self, rule, xyz, labels, pc, agl_vals):
        mask = np.ones(len(xyz), bool)
        if rule.only_unlabeled: mask &= (labels == 0)
        if rule.use_agl and agl_vals is not None:
            mask &= (agl_vals >= rule.agl_min) & (agl_vals <= rule.agl_max)
        if rule.use_z:
            mask &= (xyz[:,2] >= rule.z_min) & (xyz[:,2] <= rule.z_max)
        if rule.use_intensity and pc.intensity is not None:
            mask &= (pc.intensity >= rule.int_min) & (pc.intensity <= rule.int_max)
        if rule.use_return and pc.return_num is not None:
            if rule.return_first_only: mask &= (pc.return_num == 1)
        return mask
