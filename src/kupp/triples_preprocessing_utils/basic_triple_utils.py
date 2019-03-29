# -*- coding: utf-8 -*-


from typing import Dict, Optional, Tuple

import numpy as np


def load_triples(path):
    """Load triples saved as tab separated values and save it as a NumPy array."""
    triples = np.loadtxt(
        fname=path,
        dtype=str,
        comments='@Comment@ Subject Predicate Object',
        delimiter='\t',
    )
    return triples


def create_mappings(triples: np.ndarray) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Map entiteis and relations to ids."""
    entities = np.unique(np.ndarray.flatten(np.concatenate([triples[:, 0:1], triples[:, 2:3]])))
    relations = np.unique(np.ndarray.flatten(triples[:, 1:2]).tolist())

    entity_to_id: Dict[int, str] = {
        value: key
        for key, value in enumerate(entities)
    }

    rel_to_id: Dict[int, str] = {
        value: key
        for key, value in enumerate(relations)
    }

    return entity_to_id, rel_to_id


def map_triples_to_ids(triples: np.ndarray,
                       entity_to_id: Optional[Dict[int, str]] = None,
                       rel_to_id: Optional[Dict[int, str]] = None) -> np.ndarray:
    """Mapp entites and relations to predefined ids."""

    subject_column = np.vectorize(entity_to_id.get)(triples[:, 0:1])
    relation_column = np.vectorize(rel_to_id.get)(triples[:, 1:2])
    object_column = np.vectorize(entity_to_id.get)(triples[:, 2:3])
    triples_of_ids = np.concatenate([subject_column, relation_column, object_column], axis=1)

    triples_of_ids = np.array(triples_of_ids, dtype=np.long)
    # Note: Unique changes the order of the triples
    return np.unique(ar=triples_of_ids, axis=0), entity_to_id, rel_to_id

def get_unique_entity_pairs(triples, return_indices=False):
    """
    Extract all unique entity pairs from the triples.
    """

    subjects = triples[:, 0:1]
    objects = triples[:, 2:3]

    entity_pairs = np.concatenate([subjects, objects], axis=1)

    # idx: Indices in triples of unique pairs
    _, idx = np.unique(entity_pairs, return_index=True, axis=0)
    sorted_indices = np.sort(idx)
    # uniquoe pairs where original order of triples is preserved
    unique_pairs = entity_pairs[sorted_indices]

    if return_indices:
        return unique_pairs, sorted_indices
    return unique_pairs
