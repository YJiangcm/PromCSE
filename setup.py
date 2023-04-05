import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='promcse',
    packages=['promcse'],
    version='0.0.1',
    license='MIT',
    description='A sentence embedding tool based on PromCSE',
    author='Yuxin Jiang',
    author_email='yjiangcm@connect.ust.hk',
    url='https://github.com/YJiangcm/PromCSE',
    download_url='https://github.com/YJiangcm/PromCSE/archive/refs/tags/0.0.1.tar.gz',
    keywords=['sentence', 'embedding', 'promcse', 'nlp', 'prompt'],
    install_requires=[
        "transformers",
        "scipy",
        "datasets",
        "pandas",
        "scikit-learn",
        "prettytable",
        "gradio",
        "torch",
        "setuptools"
    ]
)
