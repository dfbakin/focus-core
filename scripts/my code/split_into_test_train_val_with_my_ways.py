from pathlib import Path

project_root = Path(".")
my_ways_to_data = Path("data/ways_to_data.txt")
dir_with_original_ways = Path("data/original_split_into_sets")

dir_for_saving = Path("data/split_into_sets_with_my_ways")
dir_for_saving.mkdir(parents=True, exist_ok=True)

map_of_files_names_by_my_ways = {}

with open(my_ways_to_data, 'r') as f:
    for line in f:
        line_list = line.strip().split()
        file_way = Path(line_list[0])
        map_of_files_names_by_my_ways[file_way.stem] = line


list_of_txt = list(dir_with_original_ways.glob("*.csv"))

print(len(list_of_txt))
map_of_sets_names_by_files_name = {}
for txt_file in list_of_txt:
    txt_name = Path(txt_file)
    set_name = txt_name.stem
    map_of_sets_names_by_files_name[set_name] = []

    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip().split()
            way_to_origin_file = Path(line[0])
            file_name_for_set = way_to_origin_file.stem

            map_of_sets_names_by_files_name[set_name].append(file_name_for_set)
    
for set_name in map_of_sets_names_by_files_name:
    file_name_ = map_of_sets_names_by_files_name[set_name]
    output_file = Path(f"{dir_for_saving}/{set_name}.csv")

    with open(output_file, "w") as f:
        for name in file_name_:
            if name in map_of_files_names_by_my_ways:
                f.write(f"{map_of_files_names_by_my_ways[name]}")
            else:
                print(f"{name} is not in your map")

