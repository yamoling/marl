import json
import os


for file in os.listdir("test_maps"):
    file = os.path.join("test_maps", file)
    with open(file, "r") as f:
        summary = json.load(f)
    map_content = summary["map_file_content"]
    target_file = file.replace(".json", "")
    with open(target_file, "w") as f:
        f.write(map_content)
    

    