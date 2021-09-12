import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qanom",
    version="0.0.2",
    author="Ayal Klein",
    author_email="ayal.s.klein@gmail.com",
    description="package for Question-Answer driven Semantic Role Labeling for Nominalizations (QANom)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleinay/QANom",
    packages=setuptools.find_packages(),
    install_requires=[
        'nltk',
        'pandas',
        'tqdm',
        'sklearn',
        'transformers==2.11.0',
        'torch==1.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)