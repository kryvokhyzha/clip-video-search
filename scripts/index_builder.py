import sys


sys.path.append(".")

import datetime
import glob
from collections import OrderedDict
from pathlib import Path

import fire
import torch
from rich import print as pprint

from src.embedder import VideoEmbedder
from src.indexer import FaissANN


def main(
    pos_text: str = "a cow",
    neg_text: str = "a chihuahua",
):
    output_structure = {}
    path_to_root = Path(__file__).parent.parent
    path_to_video_folder = path_to_root / "data" / "videos"
    path_to_output = path_to_root / "output" / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    path_to_output.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    video_paths = glob.glob(str(path_to_video_folder / "*.mp4"))
    video_idx_mapping = OrderedDict((idx, video_path) for idx, video_path in enumerate(video_paths))

    texts = [pos_text, neg_text]

    embedder = VideoEmbedder(device=device)
    indexer = FaissANN(num_clusters=0, device=device)

    video_embeddings = embedder.get_video_embeddings(list(video_idx_mapping.values()))
    text_embeddings = embedder.get_text_embeddings(texts)
    text_probs = embedder.get_text_probs(video_embeddings, text_embeddings)

    indexer.train(video_embeddings)

    output_structure["videos"] = list(video_idx_mapping.values())
    output_structure["video_embeddings shape"] = video_embeddings.shape
    output_structure["text_embeddings shape"] = text_embeddings.shape
    output_structure["Label probs"] = text_probs
    output_structure["texts"] = texts
    output_structure["distances"], output_structure["raw_indices"], output_structure["indices"] = indexer.search(
        text_embeddings, k=3, indices_mapping=video_idx_mapping
    )

    indexer.save(path=path_to_output / "flat.index")

    pprint(output_structure)


fire.Fire(main)
