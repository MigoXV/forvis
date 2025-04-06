import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision

from forvis.models.resnet18 import ResNet18


class Inferencer:

    def __init__(
        self,
        ckpt_path: Path,
        device: Union[str, torch.device] = "cpu",
        organ_dict: Optional[dict] = None,
    ):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if organ_dict is None:
            self.organ_dict = json.load(open(ckpt["cfg"]["task"]["organ_dict"]))
        else:
            self.organ_dict = organ_dict
        self.reverse_organ_dict = {v: k for k, v in self.organ_dict.items()}
        self.model = ResNet18(num_classes=len(self.organ_dict)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def infer(self, image: np.ndarray) -> Tuple[str, float]:
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0][pred].item()
        return self.reverse_organ_dict[pred], prob


if __name__ == "__main__":
    from pathlib import Path

    import imageio

    ckpt_path = Path("model-bin/checkpoint57.pt")
    image_path = Path("data-bin/test_infer/brain_1-1.jpg")
    inferencer = Inferencer(ckpt_path=ckpt_path, device="cpu")
    image = imageio.imread(image_path)
    organ, prob = inferencer.infer(image)
    print(f"Image:{image_path.name} Organ: {organ}, Probability: {prob}")
