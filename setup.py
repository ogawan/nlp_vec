from setuptools import setup, find_packages

# Insert to directory
# python -m pip install --editable /work/nlp_vec

setuptools.setup(
    name="nlp_vec",
    packages=find_packages(where='nlp_vec'),
    entry_points={
        'console_scripts': [
            'nlp_vec=nlp_vec.__main__:main',
        ]
    },
)