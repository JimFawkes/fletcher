import numpy as np
import pandas as pd
import pickle
import spacy
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.summarization import keywords
from loguru import logger

_log_file_name = __file__.split("/")[-1].split(".")[0]
logger.add(f"logs/{_log_file_name}.log", rotation="1 day")
logger.add(f"logs/info_{_log_file_name}.log", rotation="1 day", level="INFO")
logger.add(f"logs/error_{_log_file_name}.log", rotation="1 day", level="ERROR")

# These sentences should be excluded, because they are not part of the actual article.
exclude_sentences = [
    "For us to continue writing great stories, we need to display ads.",
    "Please select the extension that is blocking ads.",
    "Please follow the steps below,",
    "I want to receive updates from partners and sponsors.",
    "This article is part of a feature we also send out via email as The Atlantic Daily, a newsletter with stories, ideas, and images from The Atlantic, written specially for subscribers."
    "This article is part of a feature we also send out via email as The Edge, a daily roundup of events and ideas in American politics written specially for newsletter subscribers."
    "To sign up, please enter your email address in the field provided here.",
    "Share your own thoughts via hello@theatlantic. com.",
    "(CNN)",
    "(AP)",
    "These books are books contributed by the community.",
    "Click here to contribute your book!",
    "For more information and   please see archive.",
    ". php#Texts_and_Books, Uploaders, please note:  Archive. org supports metadata about items in just about any language so long as the characters are UTF8 encoded,",
    "Click here to join Todd’s American Dispatch: a   for conservatives!",
    "Copyright 2015 The Associated Press. All rights reserved.",
    "Copyright 2016 The Associated Press. All rights reserved.",
    "Copyright 2017 The Associated Press. All rights reserved.",
    "This material may not be published, broadcast, rewritten or redistributed.",
]


@logger.catch
def clean_sent(doc, tag="", relevant_entities=["ORG", "PERSON"]):
    """Return a generator iterating over sentences.
    doc: spacy_doc
    tag: tag to pass to TaggedDocument
    """
    if not tag:
        # Set a default tag (Use the first 5 words of the doc)
        default_tag = "_".join([word.lower_ for word in doc[:5]])
        tag = default_tag.lower()
    clean_doc = []
    for sentence_idx, sent in enumerate(doc.sents):
        # logger.debug(f"Processing sentence: {sent}")
        ents = []
        current_ent = False
        clean_words = []
        for word in sent:

            # Merge all words for a single entity
            if word.ent_type_ in relevant_entities:
                if (word.ent_iob_ == "B" and not current_ent) or (word.ent_iob_ == "I"):
                    # We are now handling a new entity (previously there was no entity).
                    current_ent = True
                elif word.ent_iob_ == "B" and current_ent:
                    # A new entity is following. We need to save the previous one and start the new one
                    word_ent = "_".join(ents)
                    if word_ent[-3:] == "_’s":
                        word_ent = word_ent[:-3]
                    ents = []
                    clean_words.append(word_ent)

                else:
                    logger.error(
                        f"UNKNOWN STATE: WORD: {word}, SENT: {sent}, SENT_ID: {sentence_idx}, TAG: {tag}"
                    )
                    continue

                if (word.is_alpha or word.is_digit) and not word.is_stop:
                    ents.append(word.lower_)
                continue

            elif current_ent:
                # We are not looking at an entity but did not save the previous one
                word_ent = "_".join(ents)
                if word_ent[-3:] == "_’s":
                    word_ent = word_ent[:-3]
                current_ent = False
                ents = []
                clean_words.append(word_ent)

            if (word.is_alpha or word.is_digit) and not word.is_stop:
                clean_words.append(word.lower_)

        if not clean_words:
            # Sort out empty lists
            continue

        # Return the sentence tag and word list
        yield (tag + "_" + str(sentence_idx), clean_words)


def get_clean_sentence_form_raw_text(raw_text, nlp, tag="", exclude_strings=[]):
    """Given the raw_text, return clean_sentences.

    """
    for exclude in exclude_strings:
        raw_text = raw_text.replace(exclude, "")
    logger.info(f"Processed exclude_strings")
    doc = nlp(raw_text)
    logger.info(f"Ran spacy pipeline.")
    return clean_sent(doc, tag)


def get_tag_and_text(df):
    """Given a dataframe, generate the article tag, meta data and raw_text.

    Return tag, raw_text, meta
    """

    for title, pub, date, raw_text, author, url in zip(
        df.title, df.publication, df.date, df.content, df.author, df.url
    ):
        try:
            tag = f"{title} {pub} {date}"
            tag = tag.strip().replace(" ", "_").replace("-", "_").lower()
            meta = {
                "title": title,
                "publication": pub,
                "date": date,
                "author": author,
                "url": url,
            }
            yield tag, raw_text, meta
        except TypeError as e:
            logger.error(f"Found an error: {title} {pub}")
            logger.exception(e)


def get_dataframe(filename="data/articles_df.pkl"):
    with open(filename, "rb") as fp:
        df = pickle.load(fp)
    return df


def get_nlp(model_name="en"):
    nlp = spacy.load("en")
    return nlp


def get_data_map(filename="data/data_map.pkl"):
    with open(filename, "rb") as fp:
        data_map = pickle.load(fp)
    return data_map


def get_doc2vec_model(filename="models/doc2vec_news_model.pkl"):
    with open(filename, "rb") as fp:
        doc2vec = pickle.load(fp)
    return doc2vec


def preprocess_articles(df, nlp, model_path="models/", save_temp_after=100_000):
    """Pre-Process a dataframe of articles.

    This will create two main data_structures while iterating over all articles in the df.
        - documents: which is a list of taggedDocuments
        - data_map: which is a dictionary containing all relevant data for any article.

    Text preprocessing:
        1. split the document into sentences (spacy)
        2. split the sentences into words (spacy)
        3. merge all words creating a single named entity into a single word
            - entity recognition is done by spacy
        4. Ignore all words that are not alphanumeric
        5. Ignore all stopwords
        6. Make all words lowercase

    """

    logger.info("Start preprocessing articles")

    documents = []
    data_map = {}
    for tag, raw_text, meta in get_tag_and_text(df):
        logger.info(f"Processing Tag: {tag}")
        text_map = {
            "text_tag": tag,
            "raw_text": raw_text,
            "clean_text": "",
            "sentences": {},
            "meta": meta,
            "keywords": [],
            "tagged_docs": [],
            "is_preprocessed": False,
        }
        clean_text = ""
        for sentence_tag, sentence in get_clean_sentence_form_raw_text(
            raw_text, nlp, tag, exclude_strings=exclude_sentences
        ):
            logger.debug(f"Processing sentence: {sentence_tag}")

            try:
                logger.debug("Removing empty string.")
                sentence.remove("")
            except ValueError:
                pass

            tagged_doc = TaggedDocument(words=sentence, tags=sentence_tag)
            text_map["tagged_docs"].append(tagged_doc)
            documents.append(tagged_doc)
            text_map["sentences"][sentence_tag] = sentence

            clean_text += " ".join(sentence) + ". "

            if save_temp_after == 0 or save_temp_after is None:
                continue

            if len(documents) % save_temp_after == 0:
                logger.info(
                    f"Storing docs_and_maps, count {len(documents)/save_temp_after}"
                )
                with open("data/data_map.pkl", "wb") as fp:
                    pickle.dump(data_map, fp)

        text_map["keywords"] = get_keywords(clean_text)
        text_map["clean_text"] = clean_text
        text_map["is_preprocessed"] = True
        data_map[tag] = text_map

    logger.info("Save final_data_map.pkl")
    with open("data/final_data_map.pkl", "wb") as fp:
        pickle.dump(data_map, fp)

    logger.info("Save final_tagged_docs.pkl")
    with open("data/final_tagged_docs.pkl", "wb") as fp:
        pickle.dump(documents, fp)

    return documents


def preprocess_raw_text(raw_text, nlp, model_path="models/"):
    """Pre-Process a raw_text.

    This will create two main data_structures while iterating over all articles in the df.
        - documents: which is a list of taggedDocuments
        - data_map: which is a dictionary containing all relevant data for any article.

    Text preprocessing:
        1. split the document into sentences (spacy)
        2. split the sentences into words (spacy)
        3. merge all words creating a single named entity into a single word
            - entity recognition is done by spacy
        4. Ignore all words that are not alphanumeric
        5. Ignore all stopwords
        6. Make all words lowercase

    """

    logger.info("Start preprocessing raw_text")

    text_map = {
        "text_tag": None,
        "raw_text": raw_text,
        "clean_text": "",
        "sentences": {},
        "meta": None,
        "keywords": [],
        "tagged_docs": [],
        "is_preprocessed": False,
    }
    clean_text = ""
    for sentence_tag, sentence in get_clean_sentence_form_raw_text(
        raw_text, nlp, exclude_strings=exclude_sentences
    ):
        logger.debug(f"Processing sentence: {sentence_tag}")

        try:
            logger.debug("Removing empty string.")
            sentence.remove("")
        except ValueError:
            pass

        tagged_doc = TaggedDocument(words=sentence, tags=sentence_tag)
        text_map["tagged_docs"].append(tagged_doc)
        text_map["sentences"][sentence_tag] = sentence

        clean_text += " ".join(sentence) + ". "

    if text_map["text_tag"] is None:
        text_map["text_tag"] = sentence_tag.rsplit("_", 1)[0]
    text_map["keywords"] = get_keywords(clean_text)
    text_map["clean_text"] = clean_text
    text_map["is_preprocessed"] = True

    return text_map


def load_data_map(filename="data/final_data_map.pkl"):
    """Load the data_map from a file.

    The data_map contains a dictionary of dictonaries, with an article tag linking to all info for an article.
    Examples are:
        - raw_text
        - clean_text
        - keywords
        - sentences with sentence tags
    """
    logger.info("Load data_map")

    with open(filename, "rb") as fp:
        data_map = pickle.load(fp)

    return data_map


def run_preprocessing(model_path="models/", save_temp_after=1_000_000):
    """Run preprocessing. Wrapper around preprocess_articles."""
    logger.info(f"Run Pre-Processing")
    df = get_dataframe()
    nlp = get_nlp()

    documents = preprocess_articles(
        df, nlp, model_path=model_path, save_temp_after=save_temp_after
    )

    return documents


def get_documents(filename="data/final_tagged_docs.pkl", run_from_scratch=False):
    """Get tagged documents.

    First try to load from a file, if this fails, re-run preprocessing.
    """
    logger.info(f"Get documnets")
    try:
        with open(filename, "rb") as fp:
            documents = pickle.load(fp)
            logger.info(f"Found pickled documents in {filename}")
            return documents
    except FileNotFoundError:
        logger.info(f"FileNotFoundError: {filename}")
        logger.info("Continuing with preprocessing")
        if not run_from_scratch:
            raise FileNotFoundError(
                f"Could not find {filename} and do not run from scratch."
            )

    return run_preprocessing()


def get_keywords(text, key_word_ratio=0.25, split=True, lemmatize=True, scores=False):
    """Get the top 'key_word_ratio' percent of unique keywords out of text"""
    logger.info("Get Keywords")

    # try to make the words count somehow dynamic
    return keywords(
        text, ratio=key_word_ratio, split=split, lemmatize=lemmatize, scores=scores
    )


def train_from_pickle(filename="data/new_tagged_docs.pkl"):
    logger.info("Train from Pickled file")
    docs = get_documents(filename="data/new_tagged_docs.pkl")
    logger.info("Start Training")
    model = train_model(documents=docs)
    return model


def train_model(documents=None):
    """Train Doc2Vec Model with tagged documents."""

    if not documents:
        logger.info("No documents were past.")
        documents = get_documents()

    logger.info(f"Starting to train model")
    model = Doc2Vec(
        documents,
        vector_size=10,
        window=2,
        min_count=5,
        workers=4,
        seed=42,
        epochs=25,
        negative=10,
    )

    model_filename = "models/doc2vec_news_model.pkl"
    logger.info(f"Pickled model to: {model_filename}")
    with open(model_filename, "wb") as fp:
        pickle.dump(model, fp)

    return model


@logger.catch
def train(model_path="models/", save_temp_after=100_000):
    nlp = spacy.load("en")
    logger.info(f"Start to train function")
    with open("data/articles_df.pkl", "rb") as fp:
        df = pickle.load(fp)

    documents = []
    raw_text_tag_map = {}
    sentence_tag_map = {}
    for tag, raw_text in get_tag_and_text(df):
        logger.info(f"Processing Tag: {tag}")
        raw_text_tag_map[tag] = raw_text
        for sentence_tag, sentence in get_clean_sentence_form_raw_text(
            raw_text, nlp, tag, exclude_strings=exclude_sentences
        ):
            logger.debug(f"Processing sentence: {sentence_tag}")
            documents.append(TaggedDocument(words=sentence, tags=sentence_tag))
            sentence_tag_map[sentence_tag] = sentence

            if len(documents) % save_temp_after == 0:
                logger.info(
                    f"Storing docs_and_maps, count {len(documents)/save_temp_after}"
                )
                docs_and_maps = (documents, raw_text_tag_map, sentence_tag_map)
                with open("data/temp_docs_and_maps.pkl", "wb") as fp:
                    pickle.dump(docs_and_maps, fp)

    logger.info(f"Starting to train model")
    model = Doc2Vec(
        documents,
        vector_size=5,
        window=2,
        min_count=1,
        workers=4,
        seed=42,
        epochs=50,
        negative=20,
    )

    model_filename = "models/doc2vec_news_model.pkl"
    logger.info(f"Pickled model to: {model_filename}")
    with open(model_filename, "wb") as fp:
        pickle.dump(model, fp)

    return model


if __name__ == "__main__":
    model = train()
