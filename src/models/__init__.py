from .res_unet import ResUNet

def build_model(name="res_unet_in_b2", num_classes=3, in_channels=3):
    if name != "res_unet_in_b2":
        raise ValueError(name)
    return ResUNet(num_classes=num_classes, in_channels=in_channels, base=2, norm="instance")
