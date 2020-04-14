from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='Alexander Lewzey',
    license='',
    install_requires=[
        'Click>=7.0',
        'matplotlib',
        'seaborn',
        'pandas',
        'plotly',
        'numpy',
        'pandas',
        'xlwings',
        'scikit-learn',
        'cufflinks',
        'tqdm',
        'psutil',
        'python-docx',
        'tqdm',
        'statsmodels',
        'fuzzywuzzy',
        'python-Levenshtein',
    ],
)
