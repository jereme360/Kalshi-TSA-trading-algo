"""
Setup file for TSA Prediction project.
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tsa_prediction',
    version='0.1.0',
    description='Quantitative trading system for Kalshi TSA weekly check-in contracts',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        # Data processing
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        
        # Machine Learning
        'scikit-learn>=1.0.0',
        'lightgbm>=3.3.0',
        'statsmodels>=0.13.0',
        'torch>=1.12.0',
        
        # API and Web
        'requests>=2.28.0',
        'beautifulsoup4>=4.11.0',
        'jwt>=1.3.1',
        'cryptography>=38.0.0',
        
        # Data visualization
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        
        # Utilities
        'python-dotenv>=0.20.0',
        'networkx>=2.8.0',
        'pyyaml>=6.0.0',
        'joblib>=1.1.0',
        
        # Testing
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
    ],
    extras_require={
        'dev': [
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.900',
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.7.0',
        ],
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'tsa-collect=tsa_prediction.scripts.collect:main',
            'tsa-train=tsa_prediction.scripts.train:main',
            'tsa-trade=tsa_prediction.scripts.trade:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='trading, machine learning, time series, kalshi, prediction',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/tsa_prediction/issues',
        'Source': 'https://github.com/yourusername/tsa_prediction',
    },
    include_package_data=True,
    package_data={
        'tsa_prediction': [
            'configs/*.yaml',
            'data/raw/*.csv',
            'data/processed/*.parquet',
        ],
    },
)

# Additional setup configurations
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = True
except ImportError:
    bdist_wheel = None

# If running setup.py
if __name__ == '__main__':
    try:
        setup()
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise