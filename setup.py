# setup.py

from setuptools import setup

setup(
    name='fair_sense_ai',
    version='0.8.2',
    py_modules=['fairsenseai'],
    author='Shaina Raza, Phd',
    author_email='shaina.raza@torontomu.ca',
    description='An AI-driven platform for analyzing bias in text and images.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://vectorinstitute.github.io/FairSense-AI/',  # Update with your GitHub URL
    license='Creative Commons license',  # Choose your license
    install_requires=[
        'torch',
        'transformers',
        'Pillow',
        'pandas',
        'gradio',
        'plotly',
        'pytesseract',
        'opencv-python',
        'numpy',
        # Include any other dependencies your code requires
    ],
    entry_points={
        'console_scripts': [
            'fairsenseai=fairsenseai:start_server',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update if you choose a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify your Python version
)
