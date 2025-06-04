from fairsenseai.analysis.bias import analyze_text_for_bias

# Example input text to analyze for bias
text_input = "Men are naturally better at decision-making, while women excel at emotional tasks."

# Analyze the text for bias
highlighted_text, detailed_analysis, score = analyze_text_for_bias(text_input,use_summarizer=True)

# Print the analysis results
print("Highlighted Text:", highlighted_text)
print("Detailed Analysis:", detailed_analysis)
print("Bias Score:", score)
print("\n")
