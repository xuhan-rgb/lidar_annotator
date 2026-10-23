"""Load and validate the editable parts of the SAM3 labeling recipe."""
import json
import os
from pathlib import Path


RECIPE_PATH = Path(__file__).resolve().parents[2] / "tools/autolabel/zgc_mask_recipe_v1/recipe.json"


def system_recipe_path():
    root = Path(os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')).expanduser()
    return root / 'lidar_annotator' / 'recipe.json'


def save_system_recipe(recipe):
    validate_recipe(recipe)
    path = system_recipe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path


def default_recipe():
    """Return an independent copy of the approved baseline recipe."""
    with RECIPE_PATH.open(encoding="utf-8") as f:
        recipe = json.load(f)
    recipe.setdefault("disabled_classes", [])
    return recipe


def load_recipe(path=None):
    """Load a recipe file, or the approved baseline when no path is given."""
    if path is None:
        return default_recipe()
    with Path(path).expanduser().open(encoding="utf-8") as f:
        recipe = json.load(f)
    validate_recipe(recipe)
    return recipe


def class_prompts(recipe, cid):
    cid = int(cid)
    if cid in range(8):
        return list(recipe["base_semantics"]["prompts"][str(cid)])
    if cid == 8:
        return [recipe["ground_objects"]["prompt"]]
    if cid == 9:
        roof = recipe["roof_union"]
        return [roof["roof_prompt"], *roof["canopy_prompts"]]
    raise ValueError("class ID must be between 0 and 9")


def set_class_prompts(recipe, cid, strings):
    cid = int(cid)
    prompts = [value.strip() if isinstance(value,str) else value for value in strings]
    if cid in range(8):
        recipe["base_semantics"]["prompts"][str(cid)] = prompts
    elif cid == 8:
        if len(prompts) != 1:
            raise ValueError("class 8 requires exactly one prompt")
        recipe["ground_objects"]["prompt"] = prompts[0]
    elif cid == 9:
        if len(prompts) != 4:
            raise ValueError("class 9 requires one roof prompt and three canopy prompts")
        recipe["roof_union"]["roof_prompt"] = prompts[0]
        recipe["roof_union"]["canopy_prompts"] = prompts[1:]
    else:
        raise ValueError("class ID must be between 0 and 9")


def validate_recipe(recipe):
    classes = recipe.get("raw_classes")
    if not isinstance(classes, list) or len(classes) != 10:
        raise ValueError("recipe must contain exactly ten raw classes")
    ids = [item.get("id") for item in classes]
    if sorted(ids) != list(range(10)):
        raise ValueError("raw class IDs must be exactly 0 through 9")
    names = [item.get("name", "").strip() for item in classes]
    if any(not name for name in names):
        raise ValueError("class names must be nonempty")
    disabled = recipe.get("disabled_classes", [])
    if not isinstance(disabled, list) or any(type(cid) is not int or cid not in range(10) for cid in disabled) or len(set(disabled)) != len(disabled):
        raise ValueError("disabled_classes must contain unique class IDs from 0 through 9")
    prompts_by_class = [class_prompts(recipe, cid) for cid in range(10)]
    normalized = []
    for cid, prompts in enumerate(prompts_by_class):
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"class {cid} prompts must be nonempty strings")
        values = [p.strip().casefold() for p in prompts]
        if len(values) != len(set(values)):
            raise ValueError(f"class {cid} prompts must be unique")
        normalized.extend((value, cid) for value in values)
    seen = {}
    for value, cid in normalized:
        if value in seen and seen[value] != cid:
            raise ValueError(f"prompt is shared by classes {seen[value]} and {cid}: {value}")
        seen[value] = cid
    if len(prompts_by_class[8]) != 1:
        raise ValueError("class 8 requires exactly one prompt")
    if len(prompts_by_class[9]) != 4:
        raise ValueError("class 9 requires exactly four prompts")
    from .composition_rules import validate_composition_rules
    validate_composition_rules(recipe)
    return recipe


