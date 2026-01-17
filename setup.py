from setuptools import setup, find_packages

setup(
    name='pvtools',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'matplotlib',
        'logging',
        'pathlib'
    ],
    author='Jean Allen',
    author_email='jean.allen@geog.ucsb.edu',
    description='Tools for working with Pressure-Volume curve data in python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jean-allen/pvtools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)