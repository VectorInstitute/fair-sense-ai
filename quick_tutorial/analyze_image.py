import requests
from PIL import Image
from io import BytesIO
from fairsenseai.analysis.bias import analyze_image_for_bias

# URL of the image to analyze
image_url = "https://media.top1000funds.com/wp-content/uploads/2019/12/iStock-525807555.jpg"

# Fetch and load the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Analyze the image for bias
highlighted_caption, image_analysis = analyze_image_for_bias(image, use_summarizer=True)

# Print the analysis results
print("Highlighted Caption:", highlighted_caption)
print("Image Analysis:", image_analysis)