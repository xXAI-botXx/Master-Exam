# extracts sub datasets from multiprocessing

import os
import shutil 

# Change these lines:
old_folder = "D:/Cache/nms10000_test"
new_output_folder = "D:/Cache/nms1000_test"    # the name will be advanced adjusted
search_idxs = ["0_0_2500_2500", "0_1_2500_2500", "1_0_2500_2500", "1_1_2500_2500"]
is_validation_data = False

for search_idx in search_idxs:
    old_buildings_path = f"{old_folder}/input"
    new_buildings_path = f"{new_output_folder}_{search_idx}{'_val'if is_validation_data else ''}/buildings"

    old_interpolated_path = f"{old_folder}/output"
    new_interpolated_path = f"{new_output_folder}_{search_idx}{'_val'if is_validation_data else ''}/interpolated"

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


