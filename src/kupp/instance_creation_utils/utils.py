# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np


def create_multi_label_relation_instances(unique_entity_pairs,
                                          triples: np.ndarray,
                                          num_relations,
                                          create_class_other=False
                                          ) -> Dict[tuple, np.array]:
    """Create for each (h,t) pair the multi relation label."""

    subjects = triples[:, 0:1]
    objects = triples[:, 2:3]
    relations = triples[:, 1:2]

    entity_pairs = np.concatenate([subjects, objects], axis=1)

    unique_pairs_to_multi_relations = {}

    # Create class 'other' for relations not contained in the KG
    if create_class_other:
        num_relations += 1

    for unique_pair in unique_entity_pairs:
        # Step 1: Get all relations of entity pair
        indicies = np.where((entity_pairs == unique_pair).all(-1))
        all_relations_of_pair = np.array(np.sort(relations[indicies]).tolist(), dtype=np.int)
        # Step 2: Create hot encoding labels
        multi_relation_label = np.zeros(num_relations)
        np.put(multi_relation_label, all_relations_of_pair, np.ones(len(all_relations_of_pair)))
        # Step 3: Save in dict
        unique_pairs_to_multi_relations[tuple(unique_pair)] = multi_relation_label

    return unique_pairs_to_multi_relations

def create_multi_label_objects_instance():
    pass