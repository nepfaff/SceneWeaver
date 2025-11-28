import random

class Check():
    def __init__(self, database):
        self.database = database

    def check_small_asset(self, asset_id):
        dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
        size = (dimensions["x"]*100, dimensions["y"]*100, dimensions["z"]*100)
        threshold = 25 * 25 # 25cm * 25cm is the threshold for small objects

        if size[0] * size[2] <= threshold and size[0] <= 25 and size[1] <= 25 and size[2] <= 25:
            return True, random.randint(0, 360)
        else:
            return False, 0
        
    def check_thin_asset(self, asset_id):
        dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
        twod_size = (dimensions["x"]*100, dimensions["z"]*100)
        threshold = 5 # 3cm is the threshold for thin objects # TODO: need a better way to determine thin threshold

        rotations = [0, 0, 0]
        if twod_size[0] < threshold:
            rotations = [0, 90, 0] # asset is thin in x direction
            return True, rotations

        elif twod_size[1] < threshold: 
            rotations = [90, 0, 0] # asset is thin in z direction
            return True, rotations

        else:
            return False, rotations
        
    def intersect_3d(self, box1, box2):
        # box1 and box2 are dictionaries with 'min' and 'max' keys,
        # which are tuples representing the minimum and maximum corners of the 3D box.
        for i in range(3):
            if box1['max'][i] < box2['min'][i] or box1['min'][i] > box2['max'][i]:
                return False
        return True
    
    def check_collision(self, placements):
        static_placements = placements
        # static_placements = [placement for placement in placements if placement["kinematic"] == True]

        if len(static_placements) <= 1:
            return placements
        else:
            colliding_pairs = []
            for i, placement_1 in enumerate(static_placements[:-1]):
                for placement_2 in static_placements[i+1:]:
                    box1 = self.get_bounding_box(placement_1)
                    box2 = self.get_bounding_box(placement_2)
                    if self.intersect_3d(box1, box2):
                        colliding_pairs.append((placement_1["id"], placement_2["id"]))
            id2assetId = {placement["id"]: placement["assetId"] for placement in placements}
            if len(colliding_pairs) != 0:
                remove_ids = []
                colliding_ids = list(set([pair[0] for pair in colliding_pairs] + [pair[1] for pair in colliding_pairs]))
                # order by size from small to large
                colliding_ids = sorted(colliding_ids, key=lambda x: self.database[id2assetId[x]]["assetMetadata"]["boundingBox"]["x"] * self.database[id2assetId[x]]["assetMetadata"]["boundingBox"]["z"])
                for object_id in colliding_ids:
                    remove_ids.append(object_id)
                    colliding_pairs = [pair for pair in colliding_pairs if object_id not in pair]
                    if len(colliding_pairs) == 0: break
                valid_placements = [placement for placement in placements if placement["id"] not in remove_ids]
                return valid_placements
            else:
                return placements
            
    def get_bounding_box(self, placement):
        asset_id = placement["assetId"]
        dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
        size = (dimensions["x"]*100, dimensions["y"]*100, dimensions["z"]*100)
        position = placement["position"]
        box = {"min": [position["x"]*100 - size[0]/2, position["y"]*100 - size[1]/2, position["z"]*100 - size[2]/2],
               "max": [position["x"]*100 + size[0]/2, position["y"]*100 + size[1]/2, position["z"]*100 + size[2]/2]}
        return box