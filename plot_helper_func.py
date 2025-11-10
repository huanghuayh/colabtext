import os, pickle

import numpy as np

data_root_pth="./conv2d_data/pred_plots/[3]/"

files_lst= ["bldc_2_results","bldc_5_results","bldc_6_results"]
final_lst=[]
for fil_name in files_lst:
    file_pth = f"{data_root_pth}{fil_name}"

    cur_dict=pickle.load(open(file_pth, "rb"))[0]
    result_dict = {}
    for key in cur_dict:
        cur_dict_name = key
        values = cur_dict[key]
        for val in values:
            file_name = val[5]
            predicted = val[4]
            gtruth = val[3]

            # if np.abs(predicted - gtruth) < 2.5:
            if np.abs(predicted - gtruth) < 50:
                # --- Extract distance string with 'cm' (e.g., '50cm') ---
                try:
                    parts = file_name.split("_")
                    dist_part = [p for p in parts if "cm" in p][0]
                    distance_str = dist_part  # e.g. '50cm'
                except Exception as e:
                    print(f"Could not parse distance from {file_name}: {e}")
                    continue

                # --- Add to result_dict ---
                if distance_str not in result_dict:
                    result_dict[distance_str] = [[], []]  # [[gtruths], [predicted]]

                result_dict[distance_str][0].append(gtruth)
                result_dict[distance_str][1].append(predicted)


            # Store this file’s processed dictionary
        final_lst.append((cur_dict_name,result_dict))
pickle.dump(final_lst, open(f"{data_root_pth}/aggregated_mtl_results.pkl", "wb"))
print()


