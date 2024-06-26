from setuptools import setup, find_packages

setup(
    name='irreversibility_estimator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'xgboost'
    ],
    author='Christian Bongiorno',
    author_email='christian.bongiorno@centralesupelec.fr',
    description='A package to estimate irreversibility in time series using gradient boosting classification.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/irreversibility_estimator',  # Replace with your actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
