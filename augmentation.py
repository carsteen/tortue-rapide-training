from albumentations.core.transforms_interface import ImageOnlyTransform

class TortueNormalize(ImageOnlyTransform):
    """Normalization used in Tortue Rapide:
    Divide pixel values by 255 and subtract .5

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(TortueNormalize, self).__init__(always_apply, p)
        self.p = p
        self.always_apply = always_apply

    def apply(self, image, **params):
        return image / 255.0 - .5