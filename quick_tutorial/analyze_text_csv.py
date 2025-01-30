from fairsenseai.analysis.bias import analyze_text_csv

# Loading the csv file
csv_file = open("data/text.csv", mode='r', newline='', encoding='utf-8')

# Analyze the csv for bias
html_table = analyze_text_csv(csv_file,use_summarizer=True)

# Print the analysis results
print("HTML table", html_table)