from recognition.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from config import *
import torch

def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()

def load_network(model_name, model_path=None, gpu=True):
    if model_path!=None:
        print("Loading face recognition model from ", model_path)

    if model_name == "arcface":
        net = get_model('r100', dropout=0.0, fp16=True, num_features=512)

        if gpu:
            ckpt = torch.load(ARCFACE_MODEL_PATH if model_path is None else model_path)
        else:
            ckpt = torch.load(ARCFACE_MODEL_PATH if model_path is None else model_path, map_location='cpu')
        try:
            net.load_state_dict(ckpt)
        except:
            try:
                net.load_state_dict(ckpt['net_state_dict'])
            except:
                raise RuntimeError("Unable to load " + model_name +
                                   ". Check if the especified model_path and model_name are of the same network")
    

    if gpu:
        net = net.cuda()
    return net
