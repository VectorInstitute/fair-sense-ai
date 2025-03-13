"""
Implements a risk analysis system that matches AI project descriptions
to potential risks from MIT Risk Repository and maps them to
NIST AI Risk Management Framework guidelines using embedding-based similarity search.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import traceback
import os

import faiss
import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from fairsenseai.utils.helper import style_risks


class RiskEmbeddingIndex:
    """
    A class that manages dual FAISS indexes for risk analysis and AI RMF (Risk Management Framework) mapping.

    Parameters
    ----------
    df_risk : pd.DataFrame
        DataFrame containing risk information with columns RiskID, RiskCategory, and RiskDescription
    df_ai_rmf : pd.DataFrame
        DataFrame containing AI RMF information with columns section_name, short_description, about, and suggested_actions
    faiss_index_file_risk : str, optional
        Path to the FAISS index file for risks, by default "fairsenseai/dataframes_and_indexes/risk_index.faiss"
    faiss_index_file_ai_rmf : str, optional
        Path to the FAISS index file for AI RMF, by default "fairsenseai/dataframes_and_indexes/ai_rmf_index.faiss"
    embedding_model_name : str, optional
        Name of the sentence transformer model to use as embedder, by default "all-MiniLM-L6-v2"

    Examples
    --------
    >>> risk_df = pd.DataFrame({
    ...     'RiskID': [1, 2],
    ...     'RiskCategory': ['Privacy', 'Security'],
    ...     'RiskDescription': ['Data breach', 'System vulnerability']
    ... })
    >>> rmf_df = pd.DataFrame({
    ...     'section_name': ['Data Protection', 'System Security'],
    ...     'short_description': ['Protect user data', 'Secure systems'],
    ...     'about': ['Data privacy guidelines', 'Security protocols'],
    ...     'suggested_actions': ['Encrypt data', 'Regular audits']
    ... })
    >>> index = RiskEmbeddingIndex(risk_df, rmf_df)
    """

    def __init__(
        self,
        df_risk: pd.DataFrame,
        df_ai_rmf: pd.DataFrame,
        faiss_index_file_risk: str,
        faiss_index_file_ai_rmf: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):

        self.df_risk = df_risk.reset_index(drop=True)
        self.df_ai_rmf = df_ai_rmf.reset_index(drop=True)

        self.embedder = SentenceTransformer(embedding_model_name)

        # Load FAISS indexes
        self.index_risk = faiss.read_index(faiss_index_file_risk)
        self.index_ai_rmf = faiss.read_index(faiss_index_file_ai_rmf)

        # 'd' dimension needed for reconstructing vectors
        self.dim_risk = self.index_risk.d
        self.dim_rmf = self.index_ai_rmf.d

        # Safety check: the two indexes should typically
        # have the same dimension if built from the same embedder.
        if self.dim_risk != self.dim_rmf:
            print(
                f"Warning: risk index dimension={self.dim_risk}, rmf index dimension={self.dim_rmf}."
            )

    def risk_with_ai_rmf(
        self, query: str, k_risk: int = 5, k_rmf: int = 1
    ) -> pd.DataFrame:
        """
        Retrieves similar risks and maps them to relevant AI RMF sections using embedding similarity.

        Parameters
        ----------
        query : str
            The input text to find similar risks for
        k_risk : int, optional
            Number of similar risks to retrieve, by default 5
        k_rmf : int, optional
            Number of AI RMF matches per risk to retrieve, by default 1

        Returns
        -------
        pd.DataFrame
            DataFrame containing matched risks and their corresponding AI RMF sections with columns:
            MIT RiskID, RiskCategory, RiskDescription, RMFSectionName, RMFShortDescription,
            RMFAbout, and RMFSuggestedActions

        Examples
        --------
        >>> index = RiskEmbeddingIndex(risk_df, rmf_df)
        >>> results = index.risk_with_ai_rmf(
        ...     "AI system handling personal data",
        ...     k_risk=2,
        ...     k_rmf=1
        ... )
        >>> print(results[['RiskID', 'RMFSectionName']].head())
           RiskID    RMFSectionName
        0      1    Data Protection
        1      2    System Security
        """

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index_risk.search(query_embedding, k_risk)

        results = []
        for i in indices[0]:
            # Reconstruct the embedding for row i
            risk_embedding = np.zeros((1, self.index_risk.d), dtype=np.float32)
            self.index_risk.reconstruct(int(i), risk_embedding[0])

            # Search in the AI RMF index
            _, ind = self.index_ai_rmf.search(risk_embedding, k_rmf)

            # Combine each matched row
            for rmf_idx in ind[0]:

                risk_row = self.df_risk.iloc[i]
                rmf_row = self.df_ai_rmf.iloc[rmf_idx]

                result_dict = {
                    "MIT Risk ID": risk_row.get("RiskID", None),
                    "MIT Risk Category": risk_row.get("RiskCategory", None),
                    "MIT Risk Description": risk_row.get("RiskDescription", None),
                    "NIST Subfunction": rmf_row.get("section_name", None),
                    "NIST Explanation": rmf_row.get("about", None),
                    "NIST Suggested Actions": rmf_row.get("suggested_actions", None),
                }
                results.append(result_dict)

        return pd.DataFrame(results)


def analyze_text_for_risks(
    text_input: str,
    csv_folder_path: Optional[str] = None,
    top_k_risk: Optional[int] = 5,
    top_k_ai_rmf: Optional[int] = 1,
    embedding_model_name: Optional[str] = "all-MiniLM-L6-v2",
    progress: Optional[gr.Progress] = gr.Progress(),
) -> tuple[str, str | None]:
    """
    Analyzes input text for AI-related risks and maps them to AI RMF guidelines using embedding-based similarity search.

    Parameters
    ----------
    text_input : str
        The user scenario text describing an AI project to be analyzed
    csv_folder_path: str
        The folder path to save csv file. if None the csv will not be saved. by default None
    top_k_risk : int, optional
        Number of similar risks to retrieve, by default 5
    top_k_ai_rmf : int, optional
        Number of AI RMF matches per risk to retrieve, by default 1
    embedding_model_name : str, optional
        Name of the sentence transformer model to use as embedder, by default "all-MiniLM-L6-v2"
    progress : gr.Progress, optional
        Gradio progress bar object for tracking analysis progress, by default gr.Progress()

    Returns
    -------
    Tuple[str, str]
        A tuple containing:
            highlighted_output : str
                HTML formatted string with highlighted risk entries.
            temp_csv_path : str
                Path to the saved CSV file containing detailed analysis results.
                Returns an empty string if analysis fails.

    Examples
    --------
    >>> scenario = "We're developing a facial recognition system for public spaces"
    >>> highlighted, csv_path = analyze_text_for_risks(scenario,top_k_risk=3,top_k_ai_rmf=2)
    >>> print(f"Results saved to: {csv_path}")

    Raises
    ------
    Exception
        If there's an error during the analysis process, returns error message
        and empty string as CSV path
    """

    progress(0, "Initializing risk analysis with embeddings...")
    try:
        script_dir = Path(__file__).resolve().parent
        main_dir = script_dir.parent
        data_dir = main_dir / "dataframes_and_indexes"

        df_risk_path = data_dir / "preprocessed_risks_df.csv"
        df_ai_rmf_path = data_dir / "AI_RMF_playbook.csv"
        faiss_risk_path = data_dir / "risk_index.faiss"
        faiss_ai_rmf_path = data_dir / "ai_rmf_index.faiss"

        df_risk = pd.read_csv(df_risk_path)
        df_ai_rmf = pd.read_csv(df_ai_rmf_path)

        risk_ai_rmf_index = RiskEmbeddingIndex(
            df_risk,
            df_ai_rmf,
            faiss_index_file_risk=str(faiss_risk_path),
            faiss_index_file_ai_rmf=str(faiss_ai_rmf_path),
            embedding_model_name=embedding_model_name,
        )

        time.sleep(0.2)
        progress(0.1, "Retrieving relevant risks...")

        # Retrieve top K from the embedding index
        top_risks_ai_rmf_df = risk_ai_rmf_index.risk_with_ai_rmf(
            text_input, k_risk=top_k_risk, k_rmf=top_k_ai_rmf
        )

        progress(
            0.2,
            f"Found {len(top_risks_ai_rmf_df)} relevant risks. Constructing prompt...",
        )

        csv_file_path = None
        if csv_folder_path is not None:
            csv_folder_path = Path(csv_folder_path)
            csv_file_name = f"Risk_Outcome_Matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_file_path = os.path.join(csv_folder_path, csv_file_name)
            top_risks_ai_rmf_df.to_csv(csv_file_path, index=False)
        else:
            print(
                "Not saving results to CSV because csv_folder_path is None."
            )

        risks_str = ""
        for _, row in top_risks_ai_rmf_df.iterrows():
            risk_id = row["MIT Risk ID"]
            risk_category = row["MIT Risk Category"]
            risk_desc = row["MIT Risk Description"]
            risks_str += (
                f"MIT Risk #{risk_id}: Category of [{risk_category}]  {risk_desc}\n"
            )

        progress(0.3, "Generating response from model...")
        time.sleep(1)
        progress(0.7, "Post-processing response...")

        highlighted_output = style_risks(top_risks_ai_rmf_df)

        progress(1.0, "Analysis complete.")
        return highlighted_output, csv_file_path

    except Exception as e:
        print('error occured')
        error_trace = traceback.format_exc()
        print(f"Error in analyze_text_for_risks: {e}\n{error_trace}")
        progress(1.0, "Analysis failed.")
        return f"Error: {e}", ""
