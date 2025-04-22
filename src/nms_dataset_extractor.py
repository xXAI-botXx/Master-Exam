# extracts sub datasets from multiprocessing

import os
import shutil 

search_idx = "1_1_2500_2500"

old_buildings_path = "./nms10000/input"
new_buildings_path = f"./nms10000_{search_idx}/buildings"

old_interpolated_path = "./nms10000/output"
new_interpolated_path = f"./nms10000_{search_idx}/interpolated"

# create new dirs
os.makedirs(new_buildings_path, exist_ok=False)
# os.makedirs(os.path.join(new_buildings_path, "metadata"), exist_ok=False)
os.makedirs(new_interpolated_path, exist_ok=False)

# copy pipeline
for cur_interpolated_file in os.listdir(old_interpolated_path):
    cur_interpolated_file_path = os.path.join(old_interpolated_path, cur_interpolated_file)
    if os.path.isfile(cur_interpolated_file_path) and search_idx in cur_interpolated_file:
        idx = "_".join(cur_interpolated_file.split("_")[:-2])
        cur_building_path = os.path.join(old_buildings_path, f"buildings_{idx}.png")
        # cur_meta_building_path = os.path.join(old_buildings_path, "metadata", f"buildings_metadata_{idx}.yml")
        id_ = idx.split("_")[0]

        # new paths
        new_interpolated_file_path = os.path.join(new_interpolated_path, f"{id_}_LAEQ_256.png")
        new_building_path = os.path.join(new_buildings_path, f"buildings_{id_}.png")
        # new_meta_building_path = os.path.join(new_buildings_path, "metadata", f"buildings_metadata_{id_}.yml")

        # copy
        shutil.copyfile(cur_interpolated_file_path, new_interpolated_file_path)
        shutil.copyfile(cur_building_path, new_building_path)
        # shutil.copyfile(cur_meta_building_path, new_meta_building_path)


