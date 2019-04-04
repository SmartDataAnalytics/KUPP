import multiprocessing
from multiprocessing import Pool

import numpy as np
import spacy
from typing import Dict

nlp = spacy.load('en')


def _extract_sentences(descriptions) -> list:
    """Extract first n sentence of description."""
    procssed_descriptions = []

    for description in descriptions:
        # print(descriptions)
        # print(description)

        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]

        sentences = []
        doc = nlp(str(description))
        for i, sent in enumerate(doc.sents):
            sentences.append(sent.text)
            if i == MAX_NUM_SENTENCES - 1:
                break

        sentences = " ".join(sentences)
        procssed_descriptions.append(sentences)

    return procssed_descriptions


def extract_first_n_sentences(entity_to_desc: Dict[str, str],
                              max_num_sentences: int,
                              num_processes=multiprocessing.cpu_count()) -> Dict[str, str]:
    """Extract first n sentences of description, and return mapping of entity to processed descriptions."""
    global MAX_NUM_SENTENCES
    MAX_NUM_SENTENCES = max_num_sentences

    entities = list(entity_to_desc.keys())
    desciptions = list(entity_to_desc.values())
    description_chunks = np.array_split(desciptions, num_processes)

    with Pool(num_processes) as p:
        # Order of results is same as order description_chunks
        # results is a list of lists where each sublist represent the resul of a process
        results = p.map(_extract_sentences, description_chunks)

    processed_decs = []

    for r in results:
        processed_decs += r

    entity_to_desc = dict(zip(entities, processed_decs))

    return entity_to_desc

