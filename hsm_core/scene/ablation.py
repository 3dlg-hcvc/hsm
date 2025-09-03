import json


def create_individual_scene_motifs(furniture: list[dict]) -> str:
    """
    scene_info:
    {
        "room_type": room_type,
        "large_furniture": [obj.to_dict() for obj in extra_scene_spec.large_objects],
        "existing_motifs": [str(motif) for motif in scene.get_all_motifs_from_type(ObjectType.LARGE)],
    }

    Return:
        {
            "arrangements": [
                {
                    "id": "unique arrangement identifier (be specific e.g. sofa, sofa_coffee_table, ceiling_lamp)",
                    "area_name": "name of scene motif",
                    "composition": {
                    "description": "direct and precise description of local furniture relationships without any style details  (e.g. a sofa in front of a TV stand)",
                    "furniture": [
                        {id: id1, amount: number of same furniture (integer)},
                        {id: id2, amount: number of same furniture (integer)},
                        ...
                    ],
                    "total_footprint": [width, height, depth],
                    "clearance": clearance_in_meters
                    },
                    "rationale": "explanation of arrangement functionality"
                }
            ]
        }
    """
    arrangements = []

    for idx, obj in enumerate(furniture):
        for ct in range(obj["amount"]):
            arrangements.append(
                {
                    "id": str(obj["id"]) + f"_{ct:02d}",
                    "area_name": f"{idx + ct:02d}_" + obj["name"].replace(" ", "_"),
                    "composition": {
                        "description": f"a {obj['name']}",
                        "furniture": [{"id": obj["id"], "amount": 1}],
                        "total_footprint": obj["dimensions"],
                        "clearance": obj["dimensions"][1],
                    },
                    "rationale": f"to place a {obj['name']} in the scene",
                }
            )

    return json.dumps({"arrangements": arrangements})

def create_individual_scene_motifs_with_analysis(furniture: list[dict], analysis: dict) -> str:
    return create_individual_scene_motifs(furniture)
