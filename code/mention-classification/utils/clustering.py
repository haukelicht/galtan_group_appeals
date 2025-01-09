import pandas as pd
import numpy as np

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

from sklearn.metrics import silhouette_score, silhouette_samples

from bertopic import BERTopic
from typing import Union, List, Literal, Tuple, Dict


def compute_coherece(
        model: BERTopic, 
        docs: Union[pd.Series, List[str]], 
        coherence_metric: Literal['u_mass', 'c_v', 'c_uci', 'c_npmi']='c_v',
        exclude_outlier_topic: bool=True,
    ) -> Tuple[Dict[str, Union[float, Dict[int, float]]], CoherenceModel]:
    """
    Compute coherence scores for a BERTopic model.

    Parameters:
        model (BERTopic): The BERTopic model.
        docs (Union[pd.Series, List[str]]): The documents to compute coherence on.
            Must be a pandas Series or a list of strings.
        coherence_metric (Literal['u_mass', 'c_v', 'c_uci', 'c_npmi'], optional): The coherence metric to use. 
            Allowed values are 'u_mass', 'c_v', 'c_uci', 'c_npmi' (see https://radimrehurek.com/gensim/models/coherencemodel.html)
            Defaults to 'c_v'.

    Returns:
        Tuple[Dict[str, Union[float, Dict[int, float]]], CoherenceModel]: 
           A tuple containing the coherence scores and the coherence model.
            - scores (Dict[str, Union[float, Dict[int, float]]]): The coherence scores.
                - overall (float): The overall coherence score.
                - by_topic (Dict[int, float]): The coherence score for each topic.
            - coherence_model (CoherenceModel): The coherence model object.

    """
    topic_words = [
        [word for word, _ in words] # for top-n words words in topic
        for tid, words in model.topic_representations_.items() # iterate over topics
        if (tid > -1 if exclude_outlier_topic else True)
    ]

    topics = model.topics_
    if exclude_outlier_topic:
        docs = [doc for doc, tid in zip(docs, topics) if tid > -1]
        topics = [tid for tid in topics if tid > -1]

    # extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    if isinstance(docs, list):
        docs = np.array(docs)
    cleaned_docs = model._preprocess_text(docs)
    toks = [analyzer(doc) for doc in cleaned_docs]

    topics = [tid for tid, toks_ in zip(topics, toks) if len(toks_) > 0]
    toks = [t for t in toks if len(t) > 0]

    # pre-process documents
    documents = pd.DataFrame({"Document": toks, "ID": range(len(toks)), "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': 'sum'})

    # extract features for Topic Coherence evaluation
    # words = vectorizer.get_feature_names_out()
    tokens = documents_per_topic.Document.to_list()
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # compile coherence model
    coherence_model = CoherenceModel(
        topics=topic_words, 
        texts=tokens, 
        corpus=corpus,
        dictionary=dictionary, 
        coherence=coherence_metric
    )

    # evaluate coherence
    scores = {
        'overall': coherence_model.get_coherence(),
        'by_topic': {tid: c for tid, c in enumerate(coherence_model.get_coherence_per_topic())}
    }
    
    return scores, coherence_model


def compute_silhouette_scores(model: BERTopic, seed: int=42):
    overall = silhouette_score(
        X=model.umap_model.embedding_, 
        labels=model.topics_, 
        sample_size=None, 
        random_state=seed
    )
    by_topic = silhouette_samples(X=model.umap_model.embedding_, labels=model.topics_)
    by_topic = pd.DataFrame({'topic': model.topics_, 'silhouette_score': by_topic})
    by_topic = by_topic.groupby('topic').agg(['mean', 'std'])
    # remove stacked columns
    by_topic.columns = by_topic.columns.droplevel(0)
    by_topic.reset_index(inplace=True)
    out = {
        'overall': overall,
        'by_topic': by_topic
    }
    return out