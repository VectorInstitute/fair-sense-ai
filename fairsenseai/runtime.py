import os
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
from codecarbon import OfflineEmissionsTracker
import codecarbon.core.powermetrics as codecarbon_powermetrics

import pandas as pd
import torch
import ollama
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
)

FAIRSENSE_RUNTIME = None

class FairsenseRuntime(object):
    """
    Base abstract class for runtime, containing common logic
    for model loading and usage.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, allow_filesystem_access: bool = True):
        """
        Initializes the runtime with the specified parameters.

        Parameters
        ----------
        allow_filesystem_access
            Whether to allow file system access for saving results.
        """
        self.allow_filesystem_access = allow_filesystem_access
        self.allowed_paths = []

        if self.allow_filesystem_access:
            print("Starting FairsenseRuntime with file system access.")

            # Making bias results directory and adding it to allowed paths
            self.bias_default_directory = "bias-results"
            os.makedirs(self.bias_default_directory, exist_ok=True)
            self.add_allowed_path(self.bias_default_directory)

            # Making risk results directory and adding it to allowed paths
            self.risk_default_directory = "risk-results"
            os.makedirs(self.risk_default_directory, exist_ok=True)
            self.add_allowed_path(self.risk_default_directory)
            
            # Making emissions directory and adding it to allowed paths
            self.emissions_output_dir = "./emission-monitoring"
            os.makedirs(self.emissions_output_dir, exist_ok=True)
            self.add_allowed_path(self.emissions_output_dir)

        else:
            print("Starting FairsenseRuntime without file system access.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_model_id = "Salesforce/blip-image-captioning-base"

        # Load Models
        print("Loading models...")
        try:
            # Image Captioning
            self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_id)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                self.blip_model_id
            ).to(self.device)

            # Summarizer for post-processing
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=0 if torch.cuda.is_available() else -1
            )

            print("Models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading models: {e}")

    @abstractmethod
    def predict_with_text_model(
        self,
        prompt: str,
        progress: Callable[[float, str], None] = None,
    ) -> str:
        """
        Abstract method to predict text using the underlying model.

        Parameters
        ----------
        prompt
            The input prompt for the text model.
        progress
            A callback function to report progress.

        Returns
        -------
        str
            The generated text response.
        """
        raise NotImplementedError

    def save_results_to_csv(self, df: pd.DataFrame, filename: str = "results.csv") -> Optional[str]:
        """
        Saves a pandas DataFrame to a CSV file in the specified directory.

        Parameters
        ----------
        df
            The DataFrame that is to be saved as a csv file.
        filename
            The name of the file to be saved.

        Returns
        -------
        Optional[str]
            The full file path of the saved CSV file.
        """
        if not self.allow_filesystem_access:
            print("[ERROR] Not saving results to CSV because filesystem access is not allowed.")
            return None

        file_path = os.path.join(self.bias_default_directory, filename)  # Combine directory and filename
        try:
            df.to_csv(file_path, index=False)  # Save the DataFrame as a CSV file
            return file_path  # Return the full file path for reference
        except Exception as e:
            return f"Error saving file: {e}"

    def add_allowed_path(self, path: str) -> str:
        """
        Adds a path to the list of allowed paths for file system access.

        Parameters
        -------
        path
            The path to add to allowed paths.

        Returns
        -------
        str
            The absolute path that was added.
        """
        if isinstance(path, str):
            path = os.path.abspath(path)
        self.allowed_paths.append(path)
        return path

    def get_allowed_paths(self) -> list[str]:
        """
        Returns the list of allowed paths for file system access.

        Returns
        -------
        list
            List of allowed paths.
        """
        return self.allowed_paths

class FairsenseGPURuntime(FairsenseRuntime):
    """
    GPU runtime class for Fairsense.
    Loads and runs a Hugging Face model locally on GPU.
    """

    def __init__(self, allow_filesystem_access: bool = True):
        """
        Initializes the GPU runtime with the specified parameters.
        
        Parameters
        ----------
        allow_filesystem_access
            Whether to allow file system access for saving results.
        """
        super().__init__(allow_filesystem_access=allow_filesystem_access)
        self.text_model_hf_id = "unsloth/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_hf_id)
        self.tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_model = AutoModelForCausalLM.from_pretrained(
            self.text_model_hf_id
        ).to(self.device).eval()
        self.text_model.resize_token_embeddings(len(self.tokenizer))

    def predict_with_text_model(
        self,
        prompt: str,
        progress: Callable[[float, str], None] = None,
        max_length: int = 1024,
        max_new_tokens: int = 200,  # Allow enough tokens for full response
        temperature: float = 0.7,
        num_beams: int = 3,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        early_stopping: bool = True
    ) -> str:
        """
        Generates a response using the model on GPU and tracks emissions using the CodeCarbon library. 
        Emissions data is saved to the directory specified by `emissions_output_dir`.

        
        Parameters
        ----------
        prompt:
            The input prompt for the text model.
        progress:
            An optional callback function to report progress.

        max_length:
            Maximum length of the input prompt, including special tokens.

        max_new_tokens:
            Maximum number of new tokens the model can generate in the output.

        temperature:
            Controls the randomness in sampling. Higher values (e.g., 1.0) increase randomness, 
            while lower values (e.g., 0.1) make output deterministic.

        num_beams:
            Number of beams for beam search. 
            Higher values improve exploration but increase computation time.

        repetition_penalty:
            Applied during beam search to penalize repeated sequences. Values > 1 discourage repetition.

        do_sample:
            Whether to use sampling for output generation. 
            Set to `True` for diverse outputs and `False` for deterministic outputs.

        early_stopping:
            Whether to stop beam search when the first completed sequence is generated.

        Returns
        -------
        str
            The text response generated by the model.
        """ 

        # Disabling powermetrics check on apple silicon because it requires sudo access
        # TODO: Remove this once they have replaced it or added a better way to bypass it
        # https://github.com/mlco2/codecarbon/issues/731
        codecarbon_powermetrics.is_powermetrics_available = lambda: False

        tracker = OfflineEmissionsTracker(
            country_iso_code="CAN", 
            project_name="fairsense-gpu-inference", 
            output_dir=self.emissions_output_dir,
        )
        
        tracker.start()
        try:
            if progress:
                progress(0.1, "Tokenizing prompt...")

            # Tokenize the input prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            if progress:
                progress(0.3, "Generating response...")

            # Generate output with safety checks
            with torch.no_grad():
                outputs = self.text_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    early_stopping=early_stopping
                )

            if progress:
                progress(0.7, "Decoding output...")

            # Decode the generated tokens
            start_index = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][start_index:], skip_special_tokens=True
            ).strip()

            # Handle incomplete responses
            if len(response.split()) < 3 or response.endswith("...") or response[-1] not in ".!?":
                warning_message = (
                    "<span style='color: black; font-style: italic;'>"
                    "(Note: The response is concise. Adjust `max_new_tokens` for additional details.)"
                    "</span>"
                )
                response += warning_message

            if progress:
                progress(1.0, "Done")
        
        finally:
            tracker.stop()
        
        return response


class FairsenseCPURuntime(FairsenseRuntime):
    """
    CPU runtime class for Fairsense.
    Uses Ollama to interface with local Llama-based models.
    """

    def __init__(self, allow_filesystem_access: bool = True):
        """
        Initializes the CPU runtime with the specified parameters.

        Parameters
        ----------
        allow_filesystem_access
            Whether to allow file system access for saving results.
        """
        super().__init__(allow_filesystem_access=allow_filesystem_access)
        self.text_model_hf_id = "llama3.2"  # from ollama
        self.ollama_client = ollama.Client()

    def predict_with_text_model(
        self, prompt: str, progress: Callable[[float, str], None] = None
    ) -> str:
        """
        Generates a response using Ollama for model inference and tracks emissions using the CodeCarbon library. 
        Emissions data is saved to the directory specified by `emissions_output_dir`.
        
        Parameters
        ----------
        prompt
            The input prompt for the text model.
        progress
            A callback function to report progress.
        
        Returns
        -------
        str
            The text response generated by the model.
        """

        # Disabling powermetrics check on apple silicon because it requires sudo access
        # TODO: Remove this once they have replaced it or added a better way to bypass it
        # https://github.com/mlco2/codecarbon/issues/731
        codecarbon_powermetrics.is_powermetrics_available = lambda: False

        tracker = OfflineEmissionsTracker(
            country_iso_code="CAN", 
            project_name="fairsense-cpu-inference", 
            output_dir=self.emissions_output_dir,
        )
        
        tracker.start()
        try:
            if progress:
                progress(0.1, "Preparing prompt...")

            if progress:
                progress(0.3, "Generating response...")

            # Generate response using Ollama
            response = self.ollama_client.chat(
                model=self.text_model_hf_id,
                messages=[{"role": "user", "content": prompt}],
            )

            if progress:
                progress(0.7, "Processing response...")

            generated_text = response["message"]["content"]

            if progress:
                progress(1.0, "Done")
            
        finally:
            tracker.stop()
            
        return generated_text


def get_runtime(allow_filesystem_access=True) -> FairsenseRuntime:
    """
    Initializes and returns the global FairsenseRuntime instance.
    Ensures that the runtime is only initialized once.

    Parameters
    ----------
    allow_filesystem_access
        Whether to allow file system access for the runtime.

    Returns
    -------
    FairsenseRuntime
        The FairsenseRuntime instance.
    """
    global FAIRSENSE_RUNTIME
    if FAIRSENSE_RUNTIME is None:
        if torch.cuda.is_available():
            FAIRSENSE_RUNTIME = FairsenseGPURuntime(allow_filesystem_access)
        else:
            FAIRSENSE_RUNTIME = FairsenseCPURuntime(allow_filesystem_access)
    return FAIRSENSE_RUNTIME

def generate_response_with_model(
    prompt: str,
    progress: Callable[[float, str], None] = None
) -> str:
    """
    Higher-level function that calls the configured runtime to generate a response.

    Parameters
    ----------
    prompt
        The input prompt for the text model.
    progress
        A callback function to report progress.

    Returns
    -------
    str
        Text response generated by the model.
    """
    fairsense_runtime = get_runtime()
    try:
        return fairsense_runtime.predict_with_text_model(prompt, progress)
    except Exception as e:
        if progress:
            progress(1.0, "Error occurred")
        return f"Error generating response: {e}"
