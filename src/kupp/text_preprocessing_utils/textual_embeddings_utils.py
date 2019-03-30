import multiprocessing
from multiprocessing import Pool

import numpy as np
import spacy

from typing import Dict

def extract_first_n_sentences(entity_to_desc: Dict[str,str], max_num_sentences: int) -> Dict[str,str]:
    """Extract first n sentences of description, and return mappingg of entity to processed descriptions."""
    pass