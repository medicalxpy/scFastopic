#!/usr/bin/env python3
"""
Test script: build and update a global Topic Store from two trained datasets.

Loads topic embeddings and cell-topic matrices from results/ for two datasets
and performs unbalanced OT alignment to merge the second dataset topics into
the global store. This lets you tune alignment params before wiring it into
the full training pipeline.
"""

import argparse
import glob
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local imports
from incremental import TopicStore


def _find_single_file(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer the newest
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    return None


def load_dataset_embeddings(results_dir: str, dataset: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load topic embeddings and optional cell-topic weights for a dataset.

    Returns:
        (topic_embeddings, topic_weights or None)
    """
    topic_pat = [
        os.path.join(results_dir, "topic_embedding", f"{dataset}*topic_embeddings*.pkl"),
    ]
    cell_topic_pat = [
        os.path.join(results_dir, "cell_topic", f"{dataset}*cell_topic_matrix*.pkl"),
    ]

    topic_path = _find_single_file(topic_pat)
    if topic_path is None:
        raise FileNotFoundError(f"Cannot find topic embeddings for dataset={dataset} under {results_dir}/topic_embedding")

    with open(topic_path, "rb") as f:
        topic_embeddings = pickle.load(f)
    topic_embeddings = np.asarray(topic_embeddings, dtype=np.float32)

    cell_topic_path = _find_single_file(cell_topic_pat)
    topic_weights = None
    if cell_topic_path is not None:
        with open(cell_topic_path, "rb") as f:
            cell_topic = pickle.load(f)
        cell_topic = np.asarray(cell_topic, dtype=np.float32)
        # Sum over cells -> topic weights (normalised later by aligner)
        topic_weights = cell_topic.sum(axis=0)

    return topic_embeddings, topic_weights


def incremental_update_store_via_method(
    store: TopicStore,
    *,
    dataset_name: str,
    results_dir: str,
    reg: float,
    reg_m: float,
    smoothing: float,
    min_transport_mass: float,
    min_best_ratio: float,
) -> Dict:
    return store.add_topics(
        dataset_name=dataset_name,
        results_dir=results_dir,
        reg=reg,
        reg_m=reg_m,
        smoothing=smoothing,
        min_transport_mass=min_transport_mass,
        min_best_ratio=min_best_ratio,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test incremental topic store merging for two datasets")
    p.add_argument("--results_dir", default="scFastopic/results", help="Root results directory")
    p.add_argument("--datasets", nargs="+", default=["PBMC4k", "PBMC8k"], help="Datasets to merge in order")
    p.add_argument("--store_path", default="scFastopic/results/topic_store/topic_store.pkl", help="Path to save/load TopicStore")
    # Aligner params
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--reg_m", type=float, default=10.0)
    p.add_argument("--smoothing", type=float, default=0.5, help="EMA weight when updating matched topics")
    p.add_argument("--min_transport_mass", type=float, default=1e-3)
    p.add_argument("--min_best_ratio", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load or init store
    if os.path.exists(args.store_path):
        store = TopicStore.load(args.store_path)
        print(f"Loaded TopicStore with {store.size} topics from {args.store_path}")
    else:
        store = TopicStore()
        print("Initialized empty TopicStore")

    for idx, ds in enumerate(args.datasets):
        print(f"\n=== Dataset {idx+1}/{len(args.datasets)}: {ds} ===")
        # Optional preview of shapes
        try:
            topics, topic_weights = load_dataset_embeddings(args.results_dir, ds)
            print(f"Loaded topics: {topics.shape}")
            if topic_weights is not None:
                print(f"Loaded topic weights from cell_topic: shape={topic_weights.shape}")
        except Exception as e:
            print(f"Preview load failed (will proceed via store.add_topics): {e}")

        stats = incremental_update_store_via_method(
            store,
            dataset_name=ds,
            results_dir=args.results_dir,
            reg=args.reg,
            reg_m=args.reg_m,
            smoothing=args.smoothing,
            min_transport_mass=args.min_transport_mass,
            min_best_ratio=args.min_best_ratio,
        )
        print(f"Matched pairs: {len(stats['matched'])}")
        print(f"Added topics: {len(stats['added'])}")
        print(f"Store size now: {stats['store_size']}")

    os.makedirs(os.path.dirname(args.store_path), exist_ok=True)
    store.save(args.store_path)
    print(f"\nSaved TopicStore to {args.store_path} with {store.size} topics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
