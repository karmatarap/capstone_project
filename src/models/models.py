import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def get_pretrained_model(hyper_params: dict, num_classes: int = 2):
    """ Returns a pretrained torch vision model 

        Supports resnet, resnext and efficientnet family of models
        Allows for additional layers of Batchnorm > Dropout > ReLU

        Args:
            model_name: Name of model
            num_classes: Number of classes in output

        Returns:
            pytorch model

    """
    model_name = hyper_params["pretrained_model"]
    if "resnet" in model_name or "resnext" in model_name:
        model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif "efficientnet" in model_name:
        model = EfficientNet.from_pretrained(model_name)
        num_features = model._fc.in_features
        model._fc = nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError

    if hyper_params["num_layers"] > 0:

        layers = []
        for _ in range(hyper_params["num_layers"]):
            if len(layers) == 0:
                layers.append(nn.Linear(num_features, hyper_params["hidden_size"]))

            else:
                layers.append(
                    nn.Linear(hyper_params["hidden_size"], hyper_params["hidden_size"])
                )
            layers.append(nn.BatchNorm1d(hyper_params["hidden_size"]))
            layers.append(nn.Dropout(hyper_params["dropout"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hyper_params["hidden_size"], num_classes))

        if "efficientnet" in model_name:
            model._fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Sequential(*layers)
    return model
