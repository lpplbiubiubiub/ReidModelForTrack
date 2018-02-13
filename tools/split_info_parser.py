import json


def parse_split_info(split_file, select_idx=0):
    with open(split_file, "r") as f:
        split_info = json.load(f)
    assert select_idx < len(split_info), "select index {} exceed arr".format(select_idx)
    split_info_dict = split_info[select_idx]
    return split_info_dict

if __name__ == "__main__":
    x = parse_split_info("/home/xksj/Data/lp/re-identification/cuhk03-src/cuhk03_release/detected/splits.json")
    print(x.keys())