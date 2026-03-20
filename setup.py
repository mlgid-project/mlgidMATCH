from setuptools import setup, find_packages
from pathlib import Path
import glob

PACKAGE_NAME = 'mlgidmatch'

if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        version='0.1.0dev1',
        description='Performs peak-to-structure matching of GID patterns',
        long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        author='Mikhail Romodin',
        url='https://github.com/mlgid-project/mlgidMATCH',
        packages=find_packages(),
        include_package_data=True,
        package_data={
            PACKAGE_NAME: ["ResNet18_best_model.pt"],
        },
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.24.4',
            'pymatgen>=2025.6.14',
            'pygidsim==0.1.1',
            'xrayutilities>=1.7.6',
            'torch>=2.0.0',
            'torchvision>=0.15.0'],
        extras_require={
            'dev': [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'pytest-xdist>=3.0.0',
                'pytest-mock>=3.10.0',
                'black>=23.0.0',
                'flake8>=6.0.0',
                'isort>=5.12.0',
                'mypy>=1.0.0',
                'pre-commit>=3.0.0',
            ],
            'test': [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'pytest-xdist>=3.0.0',
                'pytest-mock>=3.10.0',
            ]
        },
        # data_files=glob.glob('mlgidmatch/data/**')
    )
