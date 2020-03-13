import os
def parse_model_config(path):
    """
    @path: file path for storing the model configuration
           type: str
    @return: model configurations, type: list[dict]
    """
    # check cfg file validity
    if not path or not os.path.exists(path):
        print("Invalid Path for Loading Model Configuraton!")
    with open(path, "r") as f:
        lines = f.readlines()
    
    # parse the text to list, only saving lines for assignment
    lines = [line.split("\n")[0] for line in lines if not line.startswith(("#", "\n"))]
    type_idx = [i for i, line in enumerate(lines) if line.startswith("[")]
    config = []
    # build dict for each block and add the corresponding definitions
    for idx1, idx2 in zip(type_idx, type_idx[1:] + [len(lines)]):
        dic = {"type": lines[idx1].split("[")[-1].split("]")[0].strip()}
        if dic["type"] == "convolutional":
            dic["batch_normalize"] = 0
        for key, val in map(lambda x: x.split("="), lines[idx1 + 1: idx2]):
            dic[key.strip()] = val.strip()
        config.append(dic)
    return config


def parse_data_config(path):
    """
    @path: file path for storing the data configuration
           type: str
    @return: data configurations, type: list[dict]
    """
    data = {"gpus": ",".join(map(str, range(4))), "num_workers": "10"}
    with open(path, 'r') as f:
        lines = f.readlines()   
    lines = [line.split("\n")[0].split("=") for line in lines if not line.startswith(("#", "\n"))]
    data.update({key.strip(): val.strip() for key, val in lines})
        
    return data
