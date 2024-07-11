from setuptools import setup, find_packages

setup(
    name='time_irreversibility_estimator',
    version='0.4.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.0.0',
        'scikit-learn>=1.5.0',
        'xgboost>=2.1.0'
    ],
    author='Christian Bongiorno',
    author_email='christian.bongiorno@centralesupelec.fr',
    description='A package to estimate time irreversibility in time series using gradient boosting classification.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bongiornoc/Time-Irreversibility-Estimator', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
