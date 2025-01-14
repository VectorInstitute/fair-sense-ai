
### Issue:
In my local setup, I noticed that if I don't pull the model again (e.g., using `ollama pull llama3.2`), it throws an error:  
**Error generating response: [WinError 10061] No connection could be made because the target machine actively refused it.**  
Do we need to inform users about this? Has this issue occurred for you as well?

---

### Completed:
1. **Image Resizing:** I resized the image for proper display.  
2. **Model Performance Testing:**  
   - Tried `llama3.2-11B vision` but found it significantly slower compared to my previous approach with `llama`, `blip`, and `llama`.  
   - My goal was to replace OCR with `llama3.2` so we can use a single model for both image and text bias detection. I’ve documented this as a recipe for now but haven’t included it in the main toolkit.
3. I changed hyperparameters for temp, max token, we need to make change n give control to users
      
      ```  max_length = 512
        max_new_tokens = 200
        temperature = 0.5
        num_beams = 5
        do_sample = False
        repetition_penalty = 1.2
        early_stopping = True '''


4. **Handling Incomplete Responses**
When generating responses, there may be situations where the output appears truncated. For example, if the generated response is very short or ends with an ellipsis (`...`), it could indicate that the output is incomplete.

The following snippet illustrates how incomplete responses are detected and adjusted:

```python
# Handle incomplete responses
if len(response.split()) < 3 or response.endswith("...") or response[-1] not in ".!?":
    response += " (Warning: Response may be truncated. Consider increasing `max_new_tokens`.)"
```
4. **Prompts**

I updated the prompts. Can we provide more user control? We can supply implicit prompts but also allow users to input their own.


## TO-DO: Allow User Control (TO-DO1)
 
- **`max_length`:** Set to the model's limit (e.g., 1024 for GPT-2).  
- **`max_new_tokens`:** Start small (e.g., 100) for faster responses, increase for longer outputs.  
- **`temperature`:** Use 0.7 for creativity, or 0.1 for deterministic outputs.  
- **`num_beams`:** Use 1 for fast, creative outputs; increase (e.g., 3-5) for better quality.  
- **`repetition_penalty`:** Default is 1.2 to reduce redundant outputs.
- **prompt edit:** can we give this control to user also  

#### **TD2:** Fix the `highlight_bias` function:
The current implementation is not working. Here's the function for reference:  
```python
def highlight_bias(text: str, bias_words: List[str]) -> str:
    """
    Highlights bias words in the text using inline HTML styling.
    """
    if not bias_words:
        return f"<div>{text}</div>"
    for word in bias_words:
        text = text.replace(
            word,
            f"<span style='color: red; font-weight: bold;'>{word}</span>"
        )
    return f"<div>{text}</div>"
```
#### **TD3:** Fix the `batch processing progress bar, it is missing` 

the batch processing bar for image and text not showing a tqdm


#### **TD4:** Fix the `tabs` 
we need to group similar ones.

#### **TD5:Prompts**

- Provide more user control by allowing users to input their own prompts.
- Supply implicit prompts as a default option.

