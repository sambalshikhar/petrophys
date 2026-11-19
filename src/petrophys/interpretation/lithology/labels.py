"""Lithology groups of free-text rock names.

User lithology intervals and codebooks name rocks in many ways ("Sst.",
"shaly sandstone", "Limestone, dolomitic", "LST"). :func:`lithology_group_for_label`
reads the rock a name describes and returns the workflow's lithology group,
or None when the name is not a recognized rock or mixes rocks.
"""

import re
from typing import Collection, Optional

#: Rock words and abbreviations and the lithology group each names.
ROCK_TERMS = {
    # Sandstones and coarse clastics
    "sandstone": "sandstone", "sand": "sandstone", "sst": "sandstone", "ss": "sandstone",
    "sdst": "sandstone", "arenite": "sandstone", "greywacke": "sandstone", "graywacke": "sandstone",
    "quartzite": "sandstone", "clastic": "sandstone", "clastics": "sandstone",
    "arkose": "clastic_coarse", "conglomerate": "clastic_coarse", "cgl": "clastic_coarse",
    "breccia": "clastic_coarse", "gravel": "clastic_coarse",
    # Silt
    "siltstone": "silt", "silt": "silt", "sltst": "silt", "slst": "silt",
    # Shales and mudrocks
    "shale": "shale", "sh": "shale", "shl": "shale", "mudstone": "shale", "mdst": "shale",
    "claystone": "shale", "clst": "shale", "clay": "shale", "mud": "shale", "mudrock": "shale",
    "argillite": "shale",
    # Limestones (Dunham textures and chalk, marl)
    "limestone": "limestone", "lst": "limestone", "ls": "limestone", "lime": "limestone",
    "chalk": "limestone", "marl": "limestone", "marlstone": "limestone", "carbonate": "limestone",
    "carb": "limestone", "wackestone": "limestone", "packstone": "limestone",
    "grainstone": "limestone", "boundstone": "limestone", "rudstone": "limestone",
    "floatstone": "limestone", "calcarenite": "limestone", "calcilutite": "limestone",
    "micrite": "limestone", "oolite": "limestone", "coquina": "limestone",
    # Dolomite
    "dolomite": "dolomite", "dolostone": "dolomite", "dol": "dolomite", "dolo": "dolomite",
    # Evaporites
    "anhydrite": "anhydrite", "anh": "anhydrite", "anhy": "anhydrite",
    "gypsum": "gypsum", "gyp": "gypsum",
    "salt": "halite", "halite": "halite",
    "sylvite": "sylvinite", "sylvine": "sylvinite", "sylvinite": "sylvinite", "potash": "sylvinite",
    "carnallite": "carnallite", "polyhalite": "polyhalite", "kainite": "kainite",
    "evaporite": "evaporite",
    # Others
    "coal": "coal", "lignite": "coal", "anthracite": "coal",
    "chert": "chert", "flint": "chert",
    "igneous": "igneous", "basalt": "igneous", "granite": "igneous", "tuff": "igneous",
    "volcanic": "igneous", "volcanics": "igneous", "andesite": "igneous", "rhyolite": "igneous",
    "dolerite": "igneous", "gabbro": "igneous", "diorite": "igneous", "basement": "igneous",
}

#: Names whose words together mean something else than their last rock word.
ROCK_PHRASES = {
    "potash salt": "sylvinite",
    "clastic coarse": "clastic_coarse",
    "lime mudstone": "limestone",
    "carbonate mudstone": "limestone",
    "calcareous shale": "calcareous_shale",
    "calcareous mudstone": "calcareous_shale",
    "calcareous claystone": "calcareous_shale",
}

_MIXTURE_WORDS = {"and", "or", "interbedded", "interbeds", "interlaminated", "alternating"}


def lithology_group_for_label(label: object, known: Optional[Collection[str]] = None) -> Optional[str]:
    """The lithology group a rock name describes, or None.

    The rock is the name's main rock word: the last rock word of the name's
    first part (before a comma, semicolon or "with"), so "shaly sandstone"
    and "Sandstone, calcareous" are sandstone and "argillaceous limestone" is
    limestone. Abbreviations (Sst, SH, LST, DOL, ANH, ...) and a few phrases
    ("calcareous shale", "lime mudstone") are recognized. A name that joins
    different rocks ("sand/shale", "sandstone and shale") or names none is
    None, as is a group outside ``known`` when that is given.
    """
    if label is None:
        return None
    text = re.sub(r"[()\[\]]", " ", str(label).strip().lower())
    head = re.split(r"[,;]|\bwith\b", text, maxsplit=1)[0]
    head = re.sub(r"[^a-z/&+\s-]", " ", head)
    words = [word for word in re.split(r"[\s-]+", head) if word]
    phrase = " ".join(words)
    group = ROCK_PHRASES.get(phrase)
    if group is None:
        parts = [part for part in re.split(r"[/&+]", phrase) if part.strip()]
        rocks, every = [], set()
        for part in parts:
            tokens = part.split()
            found = [ROCK_TERMS[token] for token in tokens if token in ROCK_TERMS]
            if found:
                rocks.append(ROCK_PHRASES.get(part.strip(), found[-1]))
                every.update(found)
        mixed = len(parts) > 1 or any(word in _MIXTURE_WORDS for word in words)
        if not rocks or (mixed and len(every | set(rocks)) > 1):
            return None
        group = rocks[-1]
    if known is not None and group not in known:
        return None
    return group


def lithology_choice(lithology: object, options: Collection[str]) -> Optional[str]:
    """The option of a lithology parameter (a function's choices declared with role ``lithology``) that a rock
    takes, or None when no option names it.

    ``lithology`` is a rock name or a lithology group (:func:`lithology_group_for_label` reads it). The rock takes,
    in order: the option its name is ("shaly sandstone" is ``shaly_sandstone`` where that is an option); its own
    group, when that is an option; the matrix the workflow's porosity step uses for the group
    (``silt`` and ``shale`` take ``sandstone`` grains, ``calcareous_shale`` ``limestone``, ``chert`` ``quartz``),
    when that is an option; ``evaporite`` for an evaporite group (halite, anhydrite, gypsum and the potash salts),
    when that is an option. A name that is no recognized rock, or a rock no option names, is None: the caller says
    so and leaves the parameter at its default.

    Examples
    --------
    >>> m = ("limestone", "sandstone", "dolomite", "quartz")
    >>> [lithology_choice(x, m) for x in ("Sst", "SILT", "shaly sandstone", "calcareous_shale", "chert", "halite")]
    ['sandstone', 'sandstone', 'sandstone', 'limestone', 'quartz', None]
    >>> lithology_choice("ANHYDRITE", ("sandstone", "shale", "evaporite"))
    'evaporite'
    >>> lithology_choice("CALCAREOUS SHALE", ("sandstone", "shale", "calcareous_shale"))
    'calcareous_shale'
    >>> [lithology_choice("Shaly Sandstone", o) for o in (("sandstone", "shaly_sandstone"), m)]
    ['shaly_sandstone', 'sandstone']
    >>> lithology_choice("sand/shale", m) is None
    True
    """
    opts = {str(o).lower() for o in options}
    name = re.sub(r"[\s-]+", "_", str(lithology if lithology is not None else "").strip().lower())
    if name in opts:
        return name
    group = lithology_group_for_label(lithology)
    if group is None:
        return None
    if group in opts:
        return group
    from petrophys.workflow.dynamic_workflow import (_CURRENT_LITHOLOGY_GROUP_TO_MATRIX,
                                                     EVAPORITE_LITHOLOGY_GROUPS)
    matrix = _CURRENT_LITHOLOGY_GROUP_TO_MATRIX.get(group)
    if matrix in opts:
        return matrix
    if group in EVAPORITE_LITHOLOGY_GROUPS and "evaporite" in opts:
        return "evaporite"
    return None


__all__ = ["ROCK_PHRASES", "ROCK_TERMS", "lithology_choice", "lithology_group_for_label"]
