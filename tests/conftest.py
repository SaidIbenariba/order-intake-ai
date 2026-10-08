import hashlib

import numpy as np
import pytest

from order_intake_ai.catalog import generate_catalog
from order_intake_ai.match import Matcher
from order_intake_ai.text import normalize


def bow_encoder(texts):
    """Tiny deterministic stand-in for the sentence-transformer: hashed bag of words, L2-normalised."""
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in normalize(t).split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1, n)


@pytest.fixture(scope="session")
def catalog():
    return generate_catalog()


@pytest.fixture(scope="session")
def matcher(catalog):
    return Matcher(catalog, encoder=bow_encoder, tau=0.4, margin=0.09)
