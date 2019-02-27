import numpy as np
import pandas as pd
import pickle
import spacy
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from loguru import logger

_log_file_name = __file__.split("/")[-1].split(".")[0]
logger.add(f"logs/{_log_file_name}.log", rotation="1 day")


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
                    logger.error(f"UNKNOWN STATE: {word}")
                    raise NotImplementedError

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
    for exclude in exclude_strings:
        raw_text = raw_text.replace(exclude, "")
    doc = nlp(raw_text)
    return clean_sent(doc, tag)


def get_tag_and_text(df):
    for title, pub, date, raw_text in zip(
        df.title, df.publication, df.date, df.content
    ):
        tag = title + " " + pub + " " + date
        tag = tag.strip().replace(" ", "_").replace("-", "_").lower()
        yield tag, raw_text


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


def get_docs(filename="data/articles_df.pkl"):
    logger.info(f"get_docs(filename={filename})")
    nlp = spacy.load("en")
    # logger.debug(f"loaded english model.")
    with open(filename, "rb") as fp:
        df = pickle.load(fp)
    for content in df.content:
        # logger.debug(f"content: {content[:6]}...")
        yield nlp(content)


def get_sents(filename="data/articles_df.pkl", eod_flag=None):
    logger.info(f"get_sents(filename={filename}, eod_flag={eod_flag})")
    for doc in get_docs(filename):
        logger.info(f"doc: {doc[:10]}...")
        for sent in doc.sents:
            # logger.debug(f"sent: {sent[:6]}...")
            yield sent
        if eod_flag:
            # logger.debug(f"At EOD.")
            yield eod_flag


def get_paragraph(filename="data/articles_df.pkl", paragraph_length=5):
    logger.info(
        f"get_paragraph(filename={filename}, paragraph_length={paragraph_length})"
    )
    eod_flag = "@#> EOD <#@"
    sent_count = 0
    paragraph = ""
    for sent in get_sents(filename, eod_flag=eod_flag):

        sent_count += 1
        # logger.debug(f"sent_count: {sent_count}")

        if str(sent) != eod_flag:
            # logger.debug(f"type(sent): {type(sent)}, sent: {sent}")
            paragraph += sent.text + " "
            # # logger.debug(f"paragraph: {paragraph}")
            if sent_count % 5 == 0:
                # logger.debug(f"Yield paragraph: {paragraph}.")
                yield paragraph.strip()
                paragraph = ""
        else:
            # logger.debug(f"Found EOD Flag")
            yield paragraph.strip()
            paragraph = ""
            sent_count = 0


@logger.catch
def train_2(model_path="models/", handler=get_paragraph, **kwargs):
    model_filename = model_path + handler.__name__ + "model.pkl"
    docs_filename = model_path + handler.__name__ + "docs.pkl"
    logger.info(
        f"train(model_path={model_path}, handler={handler.__name__}, kwargs={kwargs})"
    )
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(handler(**kwargs))]
    logger.info(f"Got all docs. A total of {len(documents)}")

    logger.info(f"Pickled docs to: {docs_filename}")
    with open(docs_filename, "wb") as fp:
        pickle.dump(documents, fp)

    model = Doc2Vec(
        documents,
        vector_size=5,
        window=2,
        min_count=1,
        workers=8,
        seed=42,
        epochs=50,
        negative=20,
    )

    logger.info(f"Pickled model to: {model_filename}")
    with open(model_filename, "wb") as fp:
        pickle.dump(model, fp)

    return model


if __name__ == "__main__":
    model = train()
