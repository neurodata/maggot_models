# Data pulling script

## Master annotation
This is where I get the definition of which neurons to pull at all. 
- "mw brain paper all neurons" - every neuron will ever possibly consider

## Specific meta-annotations
These meta-annotations will create an individual True/False column in the metadata for
each annotation.
- "mw neuron groups" - a collection of loose cell types/class labels

## Boolean/group meta-annotations
These meta-annotations are grouped and turned into a single True/False column, where a 
neuron that is in ANY of the annotations within that meta-annotation get True in the new
column.
- "mw brain outputs" (outputs)
- "mw brain inputs" (inputs)
- "mw brain A1 ascending" (a1_ascending)
- "mw brain sensories" (sensory)
- "mw A1 neurons paired" (a1_paired)
    - Got the following error when trying to pull this one
        ```
        Getting annotations under mw A1 neurons paired:

        WARNING : No annotation found for "A02o_a1l Wave-1" (pymaid)
        WARNING:pymaid:No annotation found for "A02o_a1l Wave-1"
            A02o_a1l Wave-1
        WARNING : No annotation found for "A02o_a1l Wave-1" (pymaid)
        WARNING:pymaid:No annotation found for "A02o_a1l Wave-1"
        ```
- "mw A1 sensories" (a1_sensory)

## `class_1`
These are all of the annotations from `"mw_neuron_groups"` (besides those in `class_2`, 
see below).

I rename a few classes for brevity
```
class1_name_map = {
    "picky_LN": "pLN",
    "choosy_LN": "cLN",
    "broad_LN": "bLN",
    "AN_2nd_order": "AN2",
    "MN_2nd_order": "MN2",
}
```

Every annotation is saved under `"all_class_1"` but for the field `"class_1"`, I settle
conflicts according to the priority map below:
```
priority_map = {
    "MBON": 1,
    "MBIN": 1,
    "KC": 1,
    "uPN": 1,
    "tPN": 1,
    "vPN": 1,
    "mPN": 1,
    "sens": 1,
    "APL": 1,
    "LHN": 2,
    "CN": 2,
    "dVNC": 2,
    "dSEZ": 2,
    "RGN": 2,
    "dUnk": 2,
    "FBN": 3,
    "FAN": 3,
    "LHN2": 5, 
    "CN2": 6, 
    "FB2N": 3,
    "FFN": 4,
    "MN2": 3,
    "AN2": 3,
    "vtd2": 3,
    "A00c": 1,
}
```

## `class_2`
Any of the annotations from `"mw_neuron_groups"` that have `"subclass"` in the name are 
turned into a `class_2` label.

I rename a few of these for brevity: 
```
class2_name_map = {
    "appetitive": "app",
    "aversive": "av",
    "neither": "neith",
    "olfactory": "olfac",
}
```

