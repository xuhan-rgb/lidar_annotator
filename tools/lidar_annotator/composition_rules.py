"""Ordered class priority and subtraction rules shared by debug and production."""
from copy import deepcopy

import numpy as np


def default_composition_rules(recipe):
    def priority(source, targets, evidence='class', condition='ge_classes', compare=()):
        return dict(enabled=True, source=source, targets=list(targets), evidence=evidence,
                    condition=condition, compare=list(compare), minimum_score=0.)
    return dict(priorities=[
        priority(5, [4, 5, 6], compare=[6]),
        priority(6, [4, 5, 6], condition='gt_classes', compare=[5]),
        priority(1, [*range(10), 255], 'strong', compare=[2]),
        priority(2, [*range(10), 255], 'strong', 'gt_classes', [1]),
        priority(0, [0, 4, 5, 255], 'grass', compare=[4, 5]),
        priority(7, [4, 5, 6], condition='ge_current'),
    ], subtractions=[dict(enabled=True, source=recipe['roof_union']['replacement_class'],
                         subtract=[9], result='target')])


def composition_rules(recipe):
    return deepcopy(recipe.get('composition_rules', default_composition_rules(recipe)))


def validate_composition_rules(recipe):
    rules = recipe.get('composition_rules')
    if rules is None:
        return
    if not isinstance(rules, dict) or set(rules) != {'priorities', 'subtractions'}:
        raise ValueError('合成规则必须包含优先规则和类别减法')
    def ids(values, unknown=False):
        allowed = [*range(10), 255] if unknown else list(range(10))
        return isinstance(values, list) and bool(values) and all(type(v) is int and v in allowed for v in values) and len(set(values)) == len(values)
    for kind in ('priorities', 'subtractions'):
        if not isinstance(rules[kind], list):
            raise ValueError('合成规则必须为有序列表')
        for rule in rules[kind]:
            if not isinstance(rule, dict) or type(rule.get('enabled')) is not bool or type(rule.get('source')) is not int or rule['source'] not in range(10):
                raise ValueError('规则需要有效的启用状态和原类别')
            if kind == 'subtractions':
                if not ids(rule.get('subtract')) or rule['source'] in rule['subtract'] or rule.get('result') not in ('target', 'ignore'):
                    raise ValueError('减法需选择其他类别，重叠区域归入被减类别或 ignore')
            else:
                if not ids(rule.get('targets'), True) or rule.get('condition') not in ('always', 'ge_current', 'gt_current', 'ge_classes', 'gt_classes'):
                    raise ValueError('优先规则需要有效的作用类别和比较条件')
                evidence = rule.get('evidence')
                if evidence not in ('class', 'strong', 'grass') or evidence == 'strong' and rule['source'] not in (1, 2) or evidence == 'grass' and rule['source'] != 0:
                    raise ValueError('强实例证据仅用于行人/车辆，草地证据仅用于地面')
                compare = rule.get('compare')
                if rule['condition'].endswith('_classes') and not ids(compare):
                    raise ValueError('请指定用于得分比较的类别')
                if not isinstance(compare, list) or any(type(c) is not int or c not in range(10) for c in compare):
                    raise ValueError('比较类别无效')
                score = rule.get('minimum_score')
                if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('最低像素得分必须在 0 到 1 之间')


def candidate_scores(evidence, base_scores, actor_strong, recipe):
    scores = np.zeros((10, *base_scores.shape[1:]), np.float32)
    scores[:8] = base_scores
    scores[1:3] = actor_strong
    disabled = set(recipe.get('disabled_classes', []))
    if 8 not in disabled:
        scores[8] = evidence[recipe['ground_objects']['prompt']]
    if 9 not in disabled:
        roof = recipe['roof_union']
        canopy = [evidence[p] for p in roof['canopy_prompts']]
        votes = sum(x > 0 for x in canopy)
        scores[9] = np.maximum(evidence[roof['roof_prompt']],
                               np.where(votes >= roof['minimum_canopy_pixel_support'], np.maximum.reduce(canopy), 0))
        scores[9] *= base_scores[roof['replacement_class']] > 0
    for cid in disabled:
        scores[cid] = 0
    return scores


def apply_priorities(labels, confidence, base_scores, candidates, actor_strong, evidence, recipe):
    disabled = set(recipe.get('disabled_classes', []))
    for rule in composition_rules(recipe)['priorities']:
        source = rule['source']
        if not rule['enabled'] or source in disabled:
            continue
        if rule['evidence'] == 'strong':
            score = actor_strong[source - 1]
        elif rule['evidence'] == 'grass':
            score = np.maximum.reduce([evidence[p] for p in recipe['base_semantics']['grass_prompts']])
        else:
            score = base_scores[source] if source < 8 else candidates[source]
        selected = np.isin(labels, rule['targets']) & (score > 0) & (score >= rule['minimum_score'])
        condition = rule['condition']
        if condition != 'always':
            if condition.endswith('_current'):
                compare = np.take_along_axis(base_scores, np.minimum(labels, 7)[None], axis=0)[0]
                compare[labels == 255] = 0
                for cid in (8, 9):compare[labels == cid] = candidates[cid][labels == cid]
            else:
                # Strong actor rules compare strong evidence; other rules compare base evidence.
                values = base_scores if rule['evidence'] != 'strong' else candidates
                compare = np.maximum.reduce([values[c] if c < len(values) else candidates[c] for c in rule['compare']])
            selected &= score >= compare if condition.startswith('ge_') else score > compare
        labels[selected] = source
        if rule['evidence'] == 'grass':
            confidence[selected] = np.maximum(confidence[selected], score[selected])
        else:
            confidence[selected] = score[selected]
    return labels, confidence


def apply_subtractions(labels, confidence, candidates, recipe):
    disabled = set(recipe.get('disabled_classes', []))
    for rule in composition_rules(recipe)['subtractions']:
        if not rule['enabled'] or rule['source'] in disabled:
            continue
        targets = [c for c in rule['subtract'] if c not in disabled]
        if not targets:
            continue
        planes = candidates[targets]
        winning = planes.argmax(0)
        score = planes.max(0)
        selected = (labels == rule['source']) & (score > 0)
        if rule['result'] == 'ignore':
            labels[selected] = 255; confidence[selected] = 0
        else:
            labels[selected] = np.asarray(targets, np.uint8)[winning[selected]]
            confidence[selected] = score[selected]
    return labels, confidence
