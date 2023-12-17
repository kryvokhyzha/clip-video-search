import sys


sys.path.append(".")

import glob
import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.embedder import VideoEmbedder
from src.indexer import FaissANN


@st.cache_data()
def read_index(path: str | Path) -> FaissANN:
    return FaissANN().load(path=str(path))


if __name__ == "__main__":
    path_to_root = Path(__file__).parent.parent
    path_to_video_folder = path_to_root / "data" / "videos"
    path_to_output = path_to_root / "output"
    path_to_index = path_to_output / max(os.listdir(path_to_output)) / "flat.index"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    indexer = read_index(path=path_to_index)
    embedder = VideoEmbedder(device=device)

    video_paths = glob.glob(str(path_to_video_folder / "*.mp4"))
    video_idx_mapping = OrderedDict((idx, Path(video_path).name) for idx, video_path in enumerate(video_paths))

    text = st.text_input("Text to search", "a cow")
    k = st.slider("Number of results", 1, 10, 3)

    text_embeddings = embedder.get_text_embeddings([text])

    distances, raw_indices, indices = indexer.search(text_embeddings, k=k, indices_mapping=video_idx_mapping)
    st.dataframe(pd.DataFrame({"distances": distances[0], "raw_indices": raw_indices[0], "indices": indices[0]}))
