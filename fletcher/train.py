import numpy as np
import pandas as pd
import pickle
import spacy
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from loguru import logger

_log_file_name = __file__.split("/")[-1].split(".")[0]
logger.add(f"logs/{_log_file_name}.log", rotation="1 day")


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
def train(model_path="models/", handler=get_paragraph, **kwargs):
    logger.info(
        f"train(model_path={model_path}, handler={handler.__name__}, kwargs={kwargs})"
    )
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(handler(**kwargs))]
    logger.info(f"Got all docs. A total of {len(documents)}")
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

    filename = model_path + handler.__name__ + "model.pkl"
    logger.info(f"Pickled model to: {filename}")
    with open(filename, "wb") as fp:
        pickle.dump(model, fp)

    return model


if __name__ == "__main__":
    model = train()
