"""
First draft of the fletcher Pipeline.
"""
import spacy
import numpy as np
import pandas as pd
import gensim
from loguru import logger
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from pyLDAvis import sklearn as sklearnvis
from scipy.spatial.distance import pdist

# from config import Config

_log_file_name = __file__.split("/")[-1].split(".")[0]
logger.add(f"logs/{_log_file_name}.log", rotation="1 day")

# TODO: Add this to a PIPE Config Class
custom_stop_words = {"cnn", "nyt", "bi"}
stop_words = custom_stop_words | text.ENGLISH_STOP_WORDS


# Path to where the word2vec file lives
google_word_vec_file = "/Users/meilfort/Data/models/word2vec/google_news_vector/02_2019/google_news_vectors_negative_300.bin"


class OverwriteError(Exception):
    pass


class Document:
    """Representation of a single document/article."""

    def __init__(self, raw_text):
        logger.debug(f"Initializing Document")
        self.raw_text = raw_text
        self.meta_data = {}
        self._vector = None
        self._vectors = None

    def __repr__(self):
        try:
            return self.spacy_doc.doc
        except AttributeError:
            return self.raw_text

    @property
    def spacy_doc(self):
        try:
            return self._spacy_doc
        except AttributeError:
            raise AttributeError(
                "Document has no Attribute spacy_doc. Check if preprocessor ran."
            )

    @spacy_doc.setter
    def spacy_doc(self, vals):
        self._spacy_doc = vals

    @property
    def tv(self):
        try:
            return self._tv
        except AttributeError:
            raise AttributeError(
                "Document has no Attribute tv (TermVectorizer). Check if TermVectorizer ran."
            )

    @tv.setter
    def tv(self, vals):
        vectorizer, model = vals
        self._tv = {"vectorizer": vectorizer, "model": model}

    @property
    def lda(self):
        try:
            return self._lda
        except AttributeError:
            raise AttributeError(
                "Document has no Attribute lda. Check if LDAModel ran."
            )

    @lda.setter
    def lda(self, vals):
        lda, model, vis, overwrite = vals

        if hasattr(self, "_lda") and not overwrite:
            raise OverwriteError(
                f"The LDA Model was already run for {self.doc}. If this should be overwritten, pass lda_overwrite=True."
            )

        self._lda = {"lda": lda, "model": model, "vis": vis}

    @property
    def clean_text(self):
        try:
            return self._clean_text
        except AttributeError:
            raise AttributeError(
                "Document has no Attribute clean_text. Check if PreProcessor ran."
            )

    @clean_text.setter
    def clean_text(self, vals):
        vals, overwrite = vals

        if hasattr(self, "_clean_text") and not overwrite:
            raise OverwriteError(
                f"Already got clean_text for {self}. If this should be overwritten, pass overwrite=True."
            )

        if not isinstance(vals, list):
            vals = [vals]
        self._clean_text = vals

    @property
    def word2vec(self):
        try:
            return self._word2vec
        except AttributeError:
            raise AttributeError(
                "Document has no Attribute tv (TermVectorizer). Check if TermVectorizer ran."
            )

    @word2vec.setter
    def word2vec(self, vals):
        word_vectors, vector, overwrite = vals
        if hasattr(self, "_word2vec") and not overwrite:
            raise OverwriteError(
                f"Already got word2vec for {self}. If this should be overwritten, pass overwrite=True."
            )
        self._word2vec = {"word_vectors": word_vectors, "document_vector": vector}


class PreProcessor:
    """Get an article and transform it to a form which can be used by all following pipeline parts."""

    def __init__(
        self, raw_text, overwrite=False, spacy_config={"name": "en_core_web_md"}
    ):
        logger.debug(f"Initializing PreProcessor")
        self.raw_text = raw_text
        self.doc = Document(self.raw_text)
        self.overwrite = overwrite
        self._nlp = spacy.load(**spacy_config)

    def _remove_stopwords(self):
        logger.info(f"Removing stopwords")
        clean_text = remove_stopwords(self.raw_text)
        clean_text = self._nlp(clean_text)
        return clean_text

    def run(self):
        logger.info(f"Running PreProcessor")
        # TODO: Run spacy.tokenizer on clean_text
        self.doc.clean_text = self._remove_stopwords(), self.overwrite
        spacy_doc = self._nlp(self.raw_text)
        self.doc.spacy_doc = spacy_doc
        return self.doc


class TermVectorizer:
    """Calculate the TermVector for a given Document.
    """

    def __init__(
        self,
        doc,
        vectorizer=CountVectorizer,
        vectorizer_config={
            "max_features": 100,
            "ngram_range": (2, 4),
            "analyzer": "word",
        },
    ):
        logger.debug(f"Initializing TermVectorizer")

        self.doc = doc
        self.vectorizer = vectorizer(**vectorizer_config)

    def fit(self):
        logger.info(f"Fitting TermVectorizer")
        logger.debug(f"clean_text: {self.doc.clean_text}")
        model = self.vectorizer.fit_transform(self.doc.clean_text)
        return model

    def run(self):
        logger.info(f"Running TermVectorizer")
        model = self.fit()
        self.doc.tv = self.vectorizer, model
        return self.doc


class LDAModel:
    """Run Latent Dirichlet Allocation Model.

    Currently using the sklearn version.
    """

    # Add config
    def __init__(
        self,
        doc,
        lda_overwrite=False,
        create_vis=False,
        n_topics=10,
        max_iter=20,
        learning_method="online",
        batch_size=128,
        n_jobs=-1,
        **lda_config,
    ):
        logger.debug(f"Initializing LDAModel")
        # Potentially add a per model Sub-Pipline
        self.doc = doc
        self.lda_overwrite = lda_overwrite
        self.create_vis = create_vis

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            batch_size=batch_size,
            n_jobs=n_jobs,
            **lda_config,
        )

    def fit(self):
        logger.info(f"Fitting LDAModel")
        model = self.lda.fit(self.doc.tv["model"])
        return model

    def run(self):
        logger.info(f"Running LDAModel")
        lda_model = self.fit()

        if self.create_vis:
            vis = sklearnvis.prepare(
                lda_model, self.doc.tv["model"], self.doc.tv["vectorizer"]
            )
        else:
            vis = None
        # if self.doc.lda is not None and not self.lda_overwrite:
        #     # TODO: Move logic to Documents setter
        #     raise OverwriteError(
        #         f"The LDA Model was already run for {self.doc}. If this should be overwritten, pass lda_overwrite=True."
        #     )
        self.doc.lda = self.lda, lda_model, vis, self.lda_overwrite

        return self.doc


class Word2VecModel:
    """Calculate Vectors for doc."""

    def __init__(self, doc, overwrite=False):
        logger.debug(f"Initializing Word2VecModel")
        self.doc = doc
        self.text = doc.clean_text
        self.overwrite = overwrite
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            google_word_vec_file, binary=True
        )
        self.word_vectors = {}
        self.vector_dims = self.model["Code"].shape
        self.vector = np.zeros(self.vector_dims)

    def calculate_distances(self):
        logger.info(f"Calculating distances from word2vec")
        vectors = np.array(self.word_vectors.values())
        distance = pdist(vectors, metric="euclidean")
        avg_dist = distance / distance.shape[0]
        return distance, avg_dist

    def calculate_vectors(self):
        logger.info(f"Calculating vectors from word2vec")
        # Change this to use the clean_text instead of spacy_doc

        # TODO: Check if clean_text really needs to be a list.
        for token in self.doc.clean_text[0]:

            if token.is_stop:
                continue

            try:
                vector = self.model[token.text]
                self.word_vectors[token.text] = vector
                self.vector += vector
            except KeyError:
                logger.debug(f"Could not find Vector for Token: {token.text}")
                continue

        token_count = len(self.word_vectors.keys())
        self.vector /= token_count

    def run(self):
        logger.info(f"Running Word2VecModel")
        self.calculate_vectors()
        self.doc.word2vec = self.word_vectors, self.vector, self.overwrite
        return self.doc


class Pipeline:
    """Pipline to run all the steps."""

    elements_map = {
        "pre-processor": (PreProcessor, 0),
        "tv": (TermVectorizer, 1),
        "lda": (LDAModel, 2),
        "word2vec": (Word2VecModel, 3),
    }

    def __init__(self, doc, exclude=[], **config):
        logger.debug(f"Initializing Pipeline")
        self.document = doc
        self._configs = config

        element_keys = list(set(Pipeline.elements_map.keys()) - set(exclude))
        elements = sorted(
            [Pipeline.elements_map[element_key] for element_key in element_keys],
            key=lambda x: x[1],
        )
        self._elements = [element[0] for element in elements]

    def run(self):
        logger.info(f"Running Pipeline")
        """Run the pipline.
        
        TODO: Ensure correct execution order of pipline elements.
        """
        for element in self._elements:
            config = self._configs.get(element, {})
            pipe_element = element(self.document, **config)
            self.document = pipe_element.run()

        return self.document


# TODO: Move model fitting into Pipeline
# Ensure to train/fit on all documents
