from setuptools import setup, find_packages

setup(
    name="ICPDesign",
    version="1.0.0",
    packages=find_packages(),
    description="Library to perform multi constraint protein design",
    author="Larbi Zakaria Dahmani",
    author_email='ZAKARIA@pitt.edu',
    install_requires=['prody','matplotlib', 'seaborn', 'torch', 
                       'numpy', 'pandas'],
    python_requires='>=3.7'
)