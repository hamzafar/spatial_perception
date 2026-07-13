import numpy as np

from itertools import combinations
from collections import defaultdict

from scipy.optimize import linear_sum_assignment


class ObjectAssociationPipeline:

    def __init__(self):

        self.association_distance = None
        self.bearing_overlap_margin = None


    def validate(self):

        assert self.association_distance is not None


    def generate_candidates(
        self,
        world_objects
    ):

        candidates = []

        for obj1, obj2 in combinations(world_objects, 2):

            #
            # Different camera
            #

            if obj1["camera"] == obj2["camera"]:
                continue

            #
            # Same class
            #

            if obj1["class"] != obj2["class"]:
                continue

            candidates.append(
                {
                    "obj1": obj1,
                    "obj2": obj2
                }
            )

        return candidates

    def inside_corridor(
        self,
        bearing,
        center
    ):

        return abs(bearing - center) <= self.bearing_overlap_margin


    def bearing_filter(
        self,
        candidates
    ):

        filtered_candidates = []

        corridor_centers = {

            ("front", "left"): -45,
            ("front", "right"): 45,
            ("left", "rear"): -135,
            ("rear", "right"): 135

        }

        for candidate in candidates:

            obj1 = candidate["obj1"]
            obj2 = candidate["obj2"]

            camera_pair = tuple(
                sorted(
                    [
                        obj1["camera"],
                        obj2["camera"]
                    ]
                )
            )

            #
            # Skip unsupported camera pairs
            #

            if camera_pair not in corridor_centers:
                continue

            bearing1 = np.degrees(
                np.arctan2(
                    obj1["position"][1],
                    obj1["position"][0]
                )
            )

            bearing2 = np.degrees(
                np.arctan2(
                    obj2["position"][1],
                    obj2["position"][0]
                )
            )

            #
            # Average bearing of the candidate pair
            #

            average_bearing = (bearing1 + bearing2) / 2.0

            center = corridor_centers[camera_pair]

            if self.inside_corridor(
                average_bearing,
                center
            ):
                filtered_candidates.append(candidate)

        return filtered_candidates


    def group_candidates_by_camera_pair(
        self,
        candidates
    ):

        camera_groups = defaultdict(list)

        for candidate in candidates:

            obj1 = candidate["obj1"]
            obj2 = candidate["obj2"]

            camera_pair = tuple(
                sorted(
                    [
                        obj1["camera"],
                        obj2["camera"]
                    ]
                )
            )

            camera_groups[camera_pair].append(candidate)

        return camera_groups


    def build_cost_matrix(
        self,
        camera_groups
    ):

        matrices = {}

        for camera_pair, candidates in camera_groups.items():

            #
            # Unique objects
            #

            obj1_list = []
            obj2_list = []

            for candidate in candidates:

                obj1 = candidate["obj1"]
                obj2 = candidate["obj2"]

                if not any(obj1 is obj for obj in obj1_list):
                    obj1_list.append(obj1)

                if not any(obj2 is obj for obj in obj2_list):
                    obj2_list.append(obj2)


            #
            # Cost matrix
            #

            cost_matrix = np.zeros(
                (
                    len(obj1_list),
                    len(obj2_list)
                ),
                dtype=np.float32
            )

            for i, obj1 in enumerate(obj1_list):

                for j, obj2 in enumerate(obj2_list):

                    cost_matrix[i, j] = np.linalg.norm(
                        np.array(obj1["position"]) -
                        np.array(obj2["position"])
                    )

            matrices[camera_pair] = {

                "obj1": obj1_list,
                "obj2": obj2_list,
                "cost_matrix": cost_matrix

            }

        return matrices


    def hungarian_assignment(
        self,
        matrices
    ):

        assignments = {}

        for camera_pair, data in matrices.items():

            rows, cols = linear_sum_assignment(
                data["cost_matrix"]
            )

            assignments[camera_pair] = {

                "rows": rows,
                "cols": cols,
                "obj1": data["obj1"],
                "obj2": data["obj2"],
                "cost_matrix": data["cost_matrix"]

            }

        return assignments


    def merge_duplicates(
        self,
        world_objects,
        assignments
    ):

        unique_objects = []
        matched_objects = set()

        #
        # Merge matched objects
        #

        for _, assignment in assignments.items():

            for row, col in zip(
                assignment["rows"],
                assignment["cols"]
            ):

                cost = assignment["cost_matrix"][row, col]

                if cost > self.association_distance:
                    continue

                obj1 = assignment["obj1"][row]
                obj2 = assignment["obj2"][col]

                matched_objects.add(id(obj1))
                matched_objects.add(id(obj2))

                merged = {

                    "class": obj1["class"],

                    "camera": [
                        obj1["camera"],
                        obj2["camera"]
                    ],

                    "position": (
                        np.array(obj1["position"]) +
                        np.array(obj2["position"])
                    ) / 2,

                    "distance": (
                        obj1["distance"] +
                        obj2["distance"]
                    ) / 2

                }

                unique_objects.append(merged)

        #
        # Add unmatched objects
        #

        for obj in world_objects:

            if id(obj) in matched_objects:
                continue

            unique_objects.append(obj)

        return unique_objects



    def associate(
        self,
        world_objects
    ):

        candidates = self.generate_candidates(
            world_objects
        )
        
        # if len(candidates) == 0:
        #     return world_objects

        candidates = self.bearing_filter(candidates)
        # if len(candidates) == 0:
        #     return world_objects

        camera_groups = self.group_candidates_by_camera_pair(
            candidates
        )

        matrices = self.build_cost_matrix(
            camera_groups
        )
        # if len(matrices) == 0:
        #     return world_objects

        assignments = self.hungarian_assignment(
            matrices
        )
        # if len(assignments) == 0:
        #     return world_objects

        unique_objects = self.merge_duplicates(
            world_objects,
            assignments
        )
        # if len(unique_objects) == 0:
        #     return world_objects

        return unique_objects

