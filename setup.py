from setuptools import setup, find_packages

setup(
    name='fair_sense_ai',
    version='1.0.7',  # Update this version with each new release
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'fairsenseai': [
            'ui/*', 
            'dataframes_and_indexes/*',
        ]
    },
    author='Shaina Raza, PhD',
    author_email='shaina.raza@torontomu.ca',
    description='An AI-driven platform for analyzing bias in text and images.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://vectorinstitute.github.io/FairSense-AI/',
    license='Creative Commons BY-NC 4.0',  # Update to your chosen license
    install_requires=[
        'torch',
        'torchvision',
        'transformers',
        'Pillow',
        'pandas',
        'gradio',
        'plotly',
        'pytesseract',
        'opencv-python',
        'numpy',
        'sentence-transformers',
        'ollama',
        'faiss-cpu',
        'codecarbon',
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "fairsenseai=fairsenseai.app:start_server",
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',  # Update if using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
