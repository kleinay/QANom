import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("qanom/version.txt", "r") as f:
    version = f.read().strip()

setuptools.setup(
    name="qanom",
    version=version,
    author="Ayal Klein",
    author_email="ayal.s.klein@gmail.com",
    description="package for Question-Answer driven Semantic Role Labeling for Nominalizations (QANom)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleinay/QANom",
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers>=2.11.0',
        'torch>=1.4',
        'nltk',
        'pandas',
        'tqdm',
        'sklearn',
    ],
    package_data={
        "": ["qanom/resources/catvar/catvar21.signed", "qanom/resources/catvar/LICENCE.txt", "qanom/resources/catvar/README.md", 
             "qanom/resources/verb-to-nom-heuristic/nom_verb_pairs.txt"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)