import re
from typing import Final

# Ground-cover grass, as a *scatterable object* rather than a surface type.
#
# Deliberately not the string "grass": that one already exists in
# ENVIRONMENT_CATEGORIES (a CLIP prompt set) and in
# panorama_region_result._LABEL_RULES (where ADE20K's "grass" folds into
# RegionType.GROUND alongside earth/sand/dirt/snow), and both of those meanings
# are load-bearing for region typing and terrain texturing. Reusing the name
# would make every grass *region* look like an object to Panorama Asset
# Generation and Scene Generation, which filter on ENVIRONMENT_CATEGORIES.
#
# GrassCoverStage synthesizes instances of this class directly from the grass
# area it derives (see that stage); it is never produced by CLIP typing or by
# any detector, which is why it is absent from OBJECT_CATEGORIES below and only
# registered in _ALL_KNOWN. Membership in VEGETATION_CATEGORIES is what earns it
# the rest of the vegetation pipeline for free: DISTRIBUTABLE_CATEGORIES,
# ANIMATABLE_CATEGORIES, CategoryMeshRiggingStage's sway skeleton, and
# VideoObjectExtractionStage's per-bucket tracking cap.
GRASS_TUFT_CATEGORY: Final[str] = "grass_tuft"

VEGETATION_CATEGORIES: Final[frozenset[str]] = frozenset({
    "tree", "forest", "bush", "flower", "plant", GRASS_TUFT_CATEGORY,
})

# Structures that SAM2/Grounding DINO frequently fragment into multiple
# overlapping or adjacent same-category detections that are really one
# physical object (a wall split at a doorway/gap, a building segmented per
# facade/wing, a rock face broken into several boulder-looking regions, a
# fence fragmented at each post gap). ObjectInstanceRefinementStage merges
# same-category detections in this set when they are adjacent/overlapping,
# using a looser containment/touching test than a strict same-object IoU
# threshold. Deliberately disjoint from VEGETATION_CATEGORIES (the opposite
# problem -- one blob mask that should split into several instances) and from
# UNIQUE_CATEGORIES below (already collapsed to one instance by
# ObjectDistributionStage's population-scatter skip, not by geometric merge).
COALESCE_CATEGORIES: Final[frozenset[str]] = frozenset({
    "building", "skyscraper", "wall", "fence", "rock", "barrier",
})

# Free-standing objects that can legitimately overlap a COALESCE_CATEGORIES
# structure's bounding box without being part of it -- a person standing in
# front of a building, a car parked against a wall. ObjectInstanceRefinementStage's
# absorb pass drops any OTHER small detection heavily contained within a
# structural detection's box (a door or window fragment typed differently than
# its building, a spurious sub-region), since those are almost always a
# sub-part of the structure rather than a genuinely separate object.
ABSORB_EXEMPT_CATEGORIES: Final[frozenset[str]] = frozenset({
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "animal",
})

# Each category maps to CLIP text prompts; best per-category similarity is used.
OBJECT_CATEGORIES: Final[dict[str, list[str]]] = {
    "person":       ["a photo of a person", "a photo of a man or woman walking", "a photo of people"],
    "car":          ["a photo of a car", "a photo of an automobile or sedan", "a photo of a parked car"],
    "truck":        ["a photo of a truck", "a photo of a lorry or delivery van", "a photo of a pickup truck"],
    "bus":          ["a photo of a bus", "a photo of a coach or double-decker bus", "a photo of a transit bus"],
    "motorcycle":   ["a photo of a motorcycle", "a photo of a motorbike or scooter", "a photo of a moped"],
    "bicycle":      ["a photo of a bicycle", "a photo of a bike leaning against something", "a photo of a cyclist"],
    "boat":         ["a photo of a boat", "a photo of a sailboat or yacht", "a photo of a river boat or barge"],
    "ship":         ["a photo of a large ship", "a photo of a cargo vessel or cruise ship", "a photo of a ferry"],
    "aircraft":     ["a photo of an airplane", "a photo of a helicopter", "a photo of an aircraft in the sky"],
    "train":        ["a photo of a train", "a photo of a tram or subway car", "a photo of a railway carriage"],
    "animal":       ["a photo of an animal", "a photo of a dog or cat", "a photo of wildlife in nature"],
    "building":     ["a photo of a building", "a photo of a house or apartment block", "a photo of an office or commercial building"],
    "bridge":       ["a photo of a bridge", "a photo of an overpass or viaduct", "a photo of a pedestrian bridge"],
    "tower":        ["a photo of a tower", "a photo of a clock tower or bell tower", "a photo of a communication tower"],
    "skyscraper":   ["a photo of a skyscraper", "a photo of a tall modern building", "a photo of a high-rise building"],
    "monument":     ["a photo of a monument", "a photo of a statue or sculpture outdoors", "a photo of a war memorial"],
    "landmark":     ["a photo of a famous landmark", "a photo of a historic structure", "a photo of a tourist attraction"],
    "statue":       ["a photo of a statue", "a photo of a bronze or stone sculpture outdoors", "a photo of a public art statue", "a photo of an equestrian statue"],
    "fountain":     ["a photo of a fountain", "a photo of a water fountain in a plaza or park", "a photo of a decorative fountain with jets of water"],
    "gazebo":       ["a photo of a gazebo", "a photo of a park pavilion or bandstand", "a photo of an open-sided outdoor shelter in a garden"],
    "lighthouse":   ["a photo of a lighthouse", "a photo of a coastal lighthouse tower", "a photo of a striped lighthouse on a cliff or shore"],
    "wind_turbine": ["a photo of a wind turbine", "a photo of a large wind turbine on a hillside or field", "a photo of wind energy turbines"],
    "dock":         ["a photo of a dock or pier", "a photo of a wooden pier extending over water", "a photo of a boat dock or jetty"],
    "traffic_light":["a photo of a traffic light", "a photo of a traffic signal mounted on a pole", "a photo of street traffic lights at an intersection"],
    "fire_hydrant": ["a photo of a fire hydrant", "a photo of a red or yellow fire hydrant on a sidewalk", "a photo of a street fire hydrant"],
    "bus_stop":     ["a photo of a bus stop", "a photo of a bus shelter on a street", "a photo of a transit stop shelter"],
    "fence":        ["a photo of a fence", "a photo of a wooden or iron fence", "a photo of a garden fence or chain link fence", "a photo of a wrought iron railing fence"],
    "playground":   ["a photo of a playground", "a photo of children's playground equipment", "a photo of a play area with swings and slides"],
    "tree":         ["a photo of a tree", "a photo of a large isolated tree", "a photo of a palm tree or oak", "a photo of trees in a row", "a photo of a cluster of trees", "a photo of a forest of trees"],
    "forest":       ["a photo of a forest", "a photo of dense trees or woodland", "a photo of a wooded area", "a photo of a grove of trees"],
    "bush":         ["a photo of a bush or shrub", "a photo of a hedge or large shrub", "a photo of a dense leafy bush", "a photo of garden hedgerows"],
    "flower":       ["a photo of flowers", "a photo of a flower bed or floral garden", "a photo of colorful wildflowers in bloom", "a photo of a flowering plant"],
    "plant":        ["a photo of a plant", "a photo of a potted plant or decorative plant", "a photo of a small ornamental plant"],
    "rock":         ["a photo of a rock or boulder", "a photo of a decorative stone", "a photo of a large standalone rock"],
    "waterfall":    ["a photo of a waterfall", "a photo of a cascading waterfall in nature", "a photo of water falling over rocks", "a photo of a large waterfall"],
    "bench":        ["a photo of a park bench", "a photo of outdoor seating", "a photo of a wooden bench"],
    "chair":        ["a photo of a chair", "a photo of outdoor furniture", "a photo of a patio chair or stool"],
    "table":        ["a photo of a table", "a photo of a picnic table", "a photo of an outdoor dining table"],
    "sign":         ["a photo of a sign", "a photo of a road sign or signpost", "a photo of a billboard or advertisement"],
    "streetlight":  ["a photo of a streetlight", "a photo of a lamppost", "a photo of a light pole at night"],
    "trash_bin":    ["a photo of a trash bin", "a photo of a garbage can or dumpster", "a photo of a recycling bin"],
    "barrier":      ["a photo of a barrier", "a photo of a concrete road barrier", "a photo of bollards or traffic barriers"],
    "umbrella":     ["a photo of an umbrella", "a photo of a beach umbrella or parasol", "a photo of a patio sunshade"],
    "cart":         ["a photo of a cart or trolley", "a photo of a shopping cart", "a photo of a market stall cart"],
    "kiosk":        ["a photo of a kiosk or booth", "a photo of a ticket booth", "a photo of a street vendor stand"],
    "luggage":      ["a photo of luggage or a suitcase", "a photo of bags or backpacks", "a photo of a travel bag"],
    "other":        ["a photo of an unidentified object", "a miscellaneous item outdoors"],
}

# Categories that only exist where something was BUILT IN PLACE: fixed structures and
# the street furniture that comes with them. A scene showing no built environment at
# all cannot contain any of these, whatever a crop happens to look like -- which is
# the check ObjectCategoryClusteringStage applies (see _scene_admits_built).
#
# Measured on the Rainier capture, an alpine meadow whose region map types 0.01% of the
# panorama as BUILT and whose RAM++ tags are "blanket | field | flower | moon |
# moonlight | mountain | mountain landscape | mountain range | pasture | sky | snowy |
# wild | wildflower": it still placed a 6.1 m "tower", three "statues", two
# "buildings", a "fire_hydrant" and two "tables" within 25 m of the viewer. Cropping
# the panorama at each of those boxes shows a conifer every time. CLIP scores every
# crop against the whole label set and a tall narrow dark silhouette matches the
# built-structure prompts well; nothing downstream ever asks whether the scene has
# any built environment for them to belong to.
#
# Deliberately EXCLUDES vehicles (boat, car, aircraft, train...). They are manmade but
# mobile, so a boat on a wilderness lake is a real thing and "no built region" is not
# evidence against it. That leaves the Rainier "boat" -- also a conifer -- unvetoed;
# it needs a different signal than this one.
BUILT_ENVIRONMENT_CATEGORIES: Final[frozenset[str]] = frozenset({
    "building", "skyscraper", "tower", "bridge", "dock", "playground",
    "monument", "statue", "landmark", "lighthouse", "fountain", "gazebo",
    "wind_turbine",
    "bench", "chair", "table", "streetlight", "trash_bin", "sign",
    "fence", "barrier", "umbrella", "traffic_light", "fire_hydrant",
    "bus_stop", "kiosk", "cart",
})

# ── Where a class can physically be ──────────────────────────────────────────
#
# The two sets below are read by ObjectCategoryClusteringStage._region_veto, which
# asks a question neither CLIP nor BLIP can answer: is the GROUND under this crop
# the kind of surface this class can occupy? The region map answers it, and it is
# the only opinion in the chain that comes from a different model looking at
# different evidence -- ADE20K segmentation of the panorama, rather than a caption
# of a cutout.
#
# That independence is the whole point. Measured across the five sample captures,
# the failure this catches is BLIP captioning a texture crop fluently and wrongly,
# and CLIP then mapping that caption to a confident class. On Shark Fin Cove -- a
# scene of cliffs, sea stacks and surf -- 59% of crops were typed from a caption
# rather than from the image, giving "there is a piece of chocolate cake on a
# plate" -> table, "a giraffe that is flying in the sky" -> aircraft, and "a cat
# that is sitting on a pillow" -> animal. Every one carried ordinary confidence
# (median 0.95) and every one passed GroundingDINO verification, because a tight
# cutout of a rock fills its own box and the verifier will localize almost any
# label inside it. Nothing that reads the crop can tell these from real objects.
#
# Both sets are deliberately conservative: they name only what is IMPLAUSIBLE
# beyond argument, and the veto additionally requires a crop to be almost entirely
# inside the offending region (see region_veto_fraction). Replayed over all five
# captures at 0.85 they reject 2 Paris, 4 Rainier, 12 Shark Fin and 5 Iceland
# detections, and every single rejection is junk -- including four "aircraft"
# embedded in Mount Rainier's snowfield, captioned "a blurry image of a bird",
# "a shark shark that is flying" and "a picture of a dog".

# Classes that may legitimately be surrounded by open water. Everything NOT in
# here is vetoed when its box lands almost entirely on WATER.
#
# person and animal are included, and it is not an oversight: people wade, swim and
# stand in surf, and birds sit on water. Their crops in a beach capture straddle the
# waterline constantly, and the cost of being wrong about them (deleting the only
# real detections in a coastal scene) is far worse than the cost of letting a
# hallucinated one through to the shape gates downstream.
#
# `rock` is here for the same reason and was added on evidence: without it the veto
# rejected crop_77 of the Shark Fin capture, a 58x112 px sea stack standing in the
# surf, on the strength of its box being 100% water. It is a real rock, in a cove
# named for its sea stacks. Boulders and stacks in water are a defining feature of
# coastal captures, not an anomaly in them.
#
# `tree` is deliberately NOT here. A tree entirely surrounded by water is rare, and
# the common case that produces one is a REFLECTION in still water -- which is
# exactly what this veto should be catching.
WATERBORNE_CATEGORIES: Final[frozenset[str]] = frozenset({
    "boat", "ship", "bridge", "dock", "waterfall", "fountain",
    "person", "animal", "bird", "buoy", "rock",
})

# Classes that need level, buildable ground and so cannot be standing on a bare
# rock face. Vetoed when the box lands almost entirely on RegionType.TERRAIN --
# which in this taxonomy means mountain/cliff/rock, NOT ordinary ground (that is
# RegionType.GROUND, and is not tested here).
#
# Street furniture plus vehicles. Deliberately EXCLUDES the natural classes that
# genuinely live on rock: `rock` itself above all, but also tree, plant, bush,
# flower (they root in crevices and on scree), and person/animal (climbers, goats).
# Iceland is the capture that forces that restraint -- 48 of its 101 detections sit
# almost entirely on TERRAIN because the whole island is rock, and a rule that
# rejected trees and plants there would gut a legitimate scene. Restricted to these
# classes it rejects 5, all of them phantom aircraft.
LEVEL_GROUND_CATEGORIES: Final[frozenset[str]] = frozenset({
    "bench", "chair", "table", "streetlight", "trash_bin", "sign",
    "fence", "barrier", "umbrella", "traffic_light", "fire_hydrant",
    "bus_stop", "kiosk", "cart",
    "car", "bus", "truck", "motorcycle", "bicycle", "train", "aircraft",
    "boat", "ship",
})

# Objects that are intrinsically one-of-a-kind in a scene. Multiple detections of
# these types are treated as duplicates of the same object (ObjectCorrelationStage
# merges them). Distinct from DISTRIBUTABLE_CATEGORIES below, which gates the
# opposite direction -- which *non*-duplicate categories are allowed to be scattered
# as a synthesized population.
UNIQUE_CATEGORIES: Final[frozenset[str]] = frozenset({
    "monument", "statue", "landmark", "lighthouse",
    "fountain", "gazebo", "waterfall", "bridge", "tower",
})

# Street furniture and fixtures that repeat across a built-up region the same way
# vegetation repeats across a natural one -- a plaza has many benches, a street many
# lampposts and bins. Every member is permanently static (none appear in
# ANIMATABLE_CATEGORIES below), which is what makes them safe to scatter: a synthetic
# instance never reaches VideoObjectExtractionStage's SAM2 tracker, so no per-bucket
# tracking cap is needed for them the way it is for vegetation.
STATIC_CLUTTER_CATEGORIES: Final[frozenset[str]] = frozenset({
    "bench", "chair", "table", "streetlight", "trash_bin", "sign",
    "fence", "barrier", "umbrella", "traffic_light", "fire_hydrant",
    "bus_stop", "kiosk", "cart",
})

# Categories ObjectDistributionStage/DistributionSynthesisStage are allowed to treat
# as a population to scatter more of, rather than just "not one-of-a-kind". Natural
# clutter that plausibly repeats across a matching region -- vegetation, rock
# (already coalesced per-formation by ObjectInstanceRefinementStage, see
# COALESCE_CATEGORIES, so a "rock" detection here is one real rock, not a fragment)
# -- plus the static street furniture above.
# Deliberately still excludes every *animate* countable category (person, car,
# animal, boat, train, ...): two real detections of a person or a car is not evidence
# that more of them belong scattered around, and doing so anyway is what previously
# made VideoObjectExtractionStage try to SAM2-track hundreds of synthetic people/
# vehicles/animals -- ANIMATABLE_CATEGORIES has no per-bucket tracking cap outside
# VEGETATION_CATEGORIES, since a real detection there is supposed to be a genuinely
# distinct subject worth its own trajectory, not decorative fill. Keep this set and
# ANIMATABLE_CATEGORIES disjoint outside VEGETATION_CATEGORIES for that reason.
DISTRIBUTABLE_CATEGORIES: Final[frozenset[str]] = (
    VEGETATION_CATEGORIES | frozenset({"rock"}) | STATIC_CLUTTER_CATEGORIES
)

# Categories that can plausibly exhibit visible motion worth animating -- either
# wind sway (VEGETATION_CATEGORIES) or genuine rigid-body movement (person/vehicle/
# animal) or inherent motion in the subject itself (flowing water). Everything else
# (street furniture, signage, fixed infrastructure, statues/monuments, buildings,
# rock, ...) is permanently static: an "animated" clip of a park bench or fire
# hydrant would just be SAM2 mask-tracking noise, not real motion, so
# VideoObjectExtractionStage skips tracking those categories outright rather than
# paying for a video pass whose output nothing downstream needs.
ANIMATABLE_CATEGORIES: Final[frozenset[str]] = VEGETATION_CATEGORIES | frozenset({
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "boat", "ship", "aircraft", "train", "animal",
    "fountain", "waterfall",
})

_ALL_KNOWN: Final[frozenset[str]] = frozenset(OBJECT_CATEGORIES) | frozenset({
    # Synthesized, never classified -- see GRASS_TUFT_CATEGORY. Listed here so
    # CategoryFilter include/exclude lists can name it without tripping
    # validate_categories, but kept out of OBJECT_CATEGORIES so CLIP is never
    # offered it as a label for a real crop.
    GRASS_TUFT_CATEGORY,
    "sky", "clouds", "fog", "haze", "sunset", "aurora", "rainbow", "moon", "stars",
    "water", "river", "ocean", "lake", "beach", "ground", "dirt", "grass", "field",
    "sand", "snow", "ice", "mountain", "cliff", "trail", "road", "wall", "mud",
    "other_environment",
})


# Open-vocabulary detector labels that mean a known category under another name.
# Checked before the token match in normalize_category, so a phrase that would
# otherwise match nothing (or match the wrong half of itself) lands correctly.
_CATEGORY_SYNONYMS: Final[dict[str, str]] = {
    "stone": "rock",
    "boulder": "rock",
    "rock formation": "cliff",
    "stone rock formation": "cliff",
    "sea": "ocean",
    "sea water": "water",
    "cloud": "clouds",
    "star": "stars",
    "coast": "beach",
    "shoreline": "beach",
    "formation": "cliff",
    "hill": "mountain",
    "sun": "sunset",
}


def normalize_category(label: str | None) -> str | None:
    """Map a free-text detector label onto a known category name, or return it unchanged.

    ObjectTypingStage assigns every crop a label drawn from OBJECT_CATEGORIES /
    ENVIRONMENT_CATEGORIES, so anything typed by CLIP is already a member of
    _ALL_KNOWN. Detections created AFTER that stage are not: ObjectDetectionStage
    runs 17 stages later and its GroundingDINO prompts come from
    ObjectRecognitionStage's RAM tags, so its labels are free text -- and
    frequently multi-word phrases that no exact-match test can ever recognize.

    That matters because the environment filter downstream is an exact membership
    test (`obj_class in ENVIRONMENT_CATEGORIES`). Every label it cannot recognize
    is silently treated as a placeable object. Observed across five captures:

        "moon moonlight"  -> not "moon"    -> placed a 11.7 m moon in a meadow
        "cloud"           -> not "clouds"  -> placed a cloud
        "sea water"       -> not "water"   -> placed a slab of ocean
        "stone"           -> no match      -> placed an 8.4 m boulder cut from a cliff
        "person man"      -> not "person"  -> lost its size prior as well

    Matching is by whole word so a phrase resolves on any known token it contains,
    with environment winning ties: "sea water" is scenery whichever half is read,
    and a label that names scenery at all is not something to stand in the scene.
    Unrecognized labels are returned unchanged rather than forced to a guess --
    a genuinely novel object should still reach the placement gates, which judge
    it on depth, sky fraction and size rather than on its name.
    """
    if not label:
        return label
    name = label.strip().lower()
    if name in _ALL_KNOWN:
        return name
    if name in _CATEGORY_SYNONYMS:
        return _CATEGORY_SYNONYMS[name]

    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    for token in tokens:
        if token in _CATEGORY_SYNONYMS:
            return _CATEGORY_SYNONYMS[token]
    # Environment first: a phrase naming both ("sea water", "rock face") is scenery.
    for pool in (ENVIRONMENT_CATEGORIES, OBJECT_CATEGORIES):
        for token in tokens:
            if token in pool:
                return token
    return label


def validate_categories(names: list[str]) -> None:
    """Raise ValueError if any name is not a recognized object or environment category."""
    unknown = [n for n in names if n not in _ALL_KNOWN]
    if unknown:
        raise ValueError(
            f"Unrecognized category type(s): {unknown!r}. "
            f"Valid types: {sorted(_ALL_KNOWN)}"
        )


class CategoryFilter:
    """Include/exclude filter on object category labels.

    Exactly one of include or exclude may be specified. If neither is given,
    all categories are allowed.
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        if include is not None and exclude is not None:
            raise ValueError("Specify at most one of include_categories or exclude_categories")
        if include is not None:
            validate_categories(include)
        if exclude is not None:
            validate_categories(exclude)
        self._include = frozenset(include) if include is not None else None
        self._exclude = frozenset(exclude) if exclude is not None else None

    def allows(self, category: str) -> bool:
        if self._include is not None:
            return category in self._include
        if self._exclude is not None:
            return category not in self._exclude
        return True


ENVIRONMENT_CATEGORIES: Final[dict[str, list[str]]] = {
    "sky":          ["a photo of the sky", "a clear blue sky", "an overcast or cloudy sky"],
    "clouds":       ["a photo of clouds in the sky", "a photo of storm clouds", "a photo of white fluffy clouds",
                     "a photo of cumulus clouds", "a photo of cirrus clouds", "a photo of dark storm clouds",
                     "a photo of a cloudy sky", "a photo of cloud formations"],
    "fog":          ["a photo of fog", "a photo of a foggy landscape", "a photo of mist or haze", "a photo of a misty morning"],
    "haze":         ["a photo of haze", "a photo of a hazy sky", "a photo of atmospheric haze or smog"],
    "sunset":       ["a photo of a sunset", "a photo of a sunrise", "a photo of golden hour sky",
                     "a photo of colorful sky at dusk", "a photo of twilight or dawn sky"],
    "aurora":       ["a photo of an aurora", "a photo of the northern lights", "a photo of aurora borealis or australis"],
    "rainbow":      ["a photo of a rainbow", "a photo of a double rainbow", "a photo of a rainbow in the sky"],
    "moon":         ["a photo of the moon", "a photo of a full moon in the sky", "a photo of a crescent moon"],
    "stars":        ["a photo of stars in the sky", "a photo of a starry night sky", "a photo of the milky way"],
    "water":        ["a photo of water", "a photo of a body of water", "a calm water surface"],
    "river":        ["a photo of a river", "a photo of a flowing river or stream", "a photo of a riverbank"],
    "ocean":        ["a photo of the ocean", "a photo of the sea", "ocean waves on a shore"],
    "lake":         ["a photo of a lake", "a photo of a still lake or pond", "a photo of a lakeside"],
    "beach":        ["a photo of a beach", "a photo of a sandy beach shoreline", "a photo of waves breaking on a beach", "a photo of a tropical or coastal beach"],
    "ground":       ["a photo of the ground", "a photo of bare earth or soil", "a flat open ground surface"],
    "dirt":         ["a photo of dirt or soil", "a photo of bare muddy ground", "a photo of a dirt path or field"],
    "grass":        ["a photo of grass", "a photo of a grassy lawn or meadow", "a photo of green grass on the ground"],
    "field":        ["a photo of an open field or meadow", "a photo of farmland or grassland", "a photo of a grassy field stretching to the horizon", "a photo of a wildflower meadow"],
    "sand":         ["a photo of sand", "a photo of a sandy beach or desert", "a photo of sand dunes"],
    "snow":         ["a photo of snow", "a photo of a snow-covered ground", "a snowy landscape or snowfield"],
    "ice":          ["a photo of ice", "a photo of a frozen lake or river", "a photo of ice on the ground"],
    "mountain":     ["a photo of a mountain", "a photo of a mountain range or peak", "a photo of a hillside or slope"],
    "cliff":        ["a photo of a cliff face", "a photo of a rocky cliff or coastal bluff", "a photo of sheer rock face on a mountain or coastline"],
    "trail":        ["a photo of a hiking trail", "a photo of a dirt footpath through nature", "a photo of a narrow forest trail or mountain path"],
    "road":         ["a photo of a road or street", "a photo of asphalt pavement", "a photo of a highway or path"],
    "wall":         ["a photo of a wall", "a photo of a building facade or exterior wall", "a photo of a concrete or brick wall"],
    "mud":          ["a photo of mud", "a photo of muddy ground or a muddy puddle", "a photo of wet muddy soil"],
    "other_environment": ["a photo of a natural landscape", "a photo of an outdoor environment", "an environmental background"],
}
