"""
Creates and stores FAISS vector databases for MIT Risk Repository
and Nist AI Risk Management Framework data by converting text descriptions
into embeddings using a sentence transformer model for efficient similarity searching.
"""

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from fairsenseai.utils.helper import row_to_text


def build_and_save_index(
    df: pd.DataFrame,
    faiss_index_file: str = "dataframes_and_indexes/risk_index.faiss",
    model_name: str = "all-MiniLM-L6-v2",
    mode: str = "risk",
):
    """
    Builds and saves a FAISS index for fast similarity search of risk embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing risk information to be embedded
    faiss_index_file : str, optional
        Path where the FAISS index will be saved
    model_name : str, optional
        Name of the sentence transformer model to use for generating embeddings,
        by default "all-MiniLM-L6-v2"
    mode: str, optional
        The mode of row to text function whether for risks or the AI RMF playbook

    Examples
    --------
    >>> risks_df = pd.DataFrame({
    ...     'RiskID': [1, 2],
    ...     'Description': ['Privacy breach', 'Data bias']
    ... })
    >>> build_and_save_index(
    ...     risks_df,
    ...     faiss_index_file='risks.faiss',
    ...     model_name='all-MiniLM-L6-v2'
    ... )
    FAISS index saved to: risks.faiss
    """
    embedder = SentenceTransformer(model_name)
    corpus = df.apply(row_to_text, mode=mode, axis=1).tolist()
    embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, faiss_index_file)

    print(f"FAISS index saved to: {faiss_index_file}")


if __name__ == "__main__":
    df_risk = pd.read_csv("dataframes_and_indexes/preprocessed_risks_df.csv")
    df_ai_rmf = pd.read_csv("dataframes_and_indexes/AI_RMF_playbook.csv")
    df = df_ai_rmf.reset_index(drop=True)
    build_and_save_index(
        df,
        faiss_index_file="dataframes_and_indexes/ai_rmf_index.faiss",
        mode="ai_rmf",
    )
