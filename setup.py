from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

AUTHOR_NAME = 'ISSAC'
SRC_REPO = 'src'
LIST_OF_REQUIRMENTS =['streamlit']

setup(
    name= SRC_REPO,
    version='0.0.1',
    author= AUTHOR_NAME,
    author_email= 'issacvasanth0718@gmail.com',
    description= 'A simple python package to make a simple web application',
    long_description= long_description,
    long_description_content_type= 'text/markdown',
    package = [SRC_REPO],
    python_requires = '>=3.12.6',
    install_requires = LIST_OF_REQUIRMENTS,
)