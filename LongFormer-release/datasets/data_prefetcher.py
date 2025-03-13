import torch


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)

    def next(self):
        try:
            imgs, label, img_indicators = next(self.loader)
        except StopIteration:
            imgs = None
            label = None
            img_indicators = None
        return imgs, label, img_indicators
