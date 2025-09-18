from setuptools import setup, find_packages

setup(
    name='deepchallenge4-team4',
    version='1.0',
    #packages=find_packages(),  # Explicitly specify packages
    # Alternatively, specify packages manually:
    packages=find_packages(include=['deepchallenge4-team10']),
    # Or use src-layout:
    # package_dir={'': 'src'},
    # packages=find_packages('src'),
    install_requires=[  # List of dependencies
        'pyYAML',
        'wandb',
        'torch',
        'torchvision',
        'torchinfo',
        'scipy',
        'tqdm',
        'matplotlib',
        'pandas',
        'tensorboard',
        'rasterio',
        'tifffile',
        'albumentations',
        'numpy',
        'attr']
)