from typing import Final

VEGETATION_CATEGORIES: Final[frozenset[str]] = frozenset({
    "tree", "forest", "bush", "flower", "plant"
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

# Objects that are intrinsically one-of-a-kind in a scene. Multiple detections of
# these types are treated as duplicates of the same object (ObjectCorrelationStage
# merges them). Distinct from DISTRIBUTABLE_CATEGORIES below, which gates the
# opposite direction -- which *non*-duplicate categories are allowed to be scattered
# as a synthesized population.
UNIQUE_CATEGORIES: Final[frozenset[str]] = frozenset({
    "monument", "statue", "landmark", "lighthouse",
    "fountain", "gazebo", "waterfall", "bridge", "tower",
})

# Categories ObjectDistributionStage/DistributionSynthesisStage are allowed to treat
# as a population to scatter more of, rather than just "not one-of-a-kind". Natural
# clutter that plausibly repeats across a matching region -- vegetation, and rock
# (already coalesced per-formation by ObjectInstanceRefinementStage, see
# COALESCE_CATEGORIES, so a "rock" detection here is one real rock, not a fragment).
# Deliberately excludes every other countable/discrete category (person, car, animal,
# bench, fire_hydrant, ...): two real detections of a person or a car is not evidence
# that more of them belong scattered around, and doing so anyway is what previously
# made VideoObjectExtractionStage try to SAM2-track hundreds of synthetic people/
# vehicles/animals -- ANIMATABLE_CATEGORIES has no per-bucket tracking cap outside
# VEGETATION_CATEGORIES, since a real detection there is supposed to be a genuinely
# distinct subject worth its own trajectory, not decorative fill.
DISTRIBUTABLE_CATEGORIES: Final[frozenset[str]] = VEGETATION_CATEGORIES | frozenset({"rock"})

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
    "sky", "clouds", "fog", "haze", "sunset", "aurora", "rainbow", "moon", "stars",
    "water", "river", "ocean", "lake", "beach", "ground", "dirt", "grass", "field",
    "sand", "snow", "ice", "mountain", "cliff", "trail", "road", "wall", "mud",
    "other_environment",
})


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
