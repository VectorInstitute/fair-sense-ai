from fairsenseai import analyze_images_batch

# List of image paths to analyze
image_paths = [
    "data/1ab67b0d54.jpg",
    "data/460f455f9f.jpg",
]

# Analyze the images for bias
html_table = analyze_images_batch(image_paths, use_summarizer=True)

# Print the analysis results
print("HTML table", html_table)