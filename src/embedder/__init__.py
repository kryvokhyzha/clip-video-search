from pathlib import Path
from typing import List

import clip
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from tqdm import tqdm


class VideoEmbedder:
    def __init__(self, device: str = "cpu", clip_model: str = "ViT-B/32"):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=device)

    def get_video_embeddings(self, videos: List[str | Path]) -> torch.Tensor:
        video_embeddings = []
        for video_path in videos:
            with open(video_path, "rb") as f:
                vr = VideoReader(f)

            frames = []
            for frame in tqdm(vr.get_batch(vr.get_key_indices()).asnumpy()):
                frames.append(self.preprocess(Image.fromarray(np.uint8(frame))).unsqueeze(0).to(self.device))
            frames = torch.cat(frames, dim=0)

            with torch.no_grad():
                video_features = torch.mean(self.model.encode_image(frames).detach(), dim=0, keepdim=True)
                video_features /= video_features.norm(dim=-1, keepdim=True)
                video_embeddings.append(video_features)

        return torch.cat(video_embeddings, dim=0)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        tokenize_texts = clip.tokenize(texts).to(self.device)
        text_embeddings = self.model.encode_text(tokenize_texts).detach()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    @staticmethod
    def get_text_probs(video_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return (100.0 * video_embeddings @ text_embeddings.T).softmax(dim=-1)
