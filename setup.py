from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

# this function will return the list of requirements
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as r:
        requirements = r.readlines()
        requirements = [r.replace("\n","")for  r in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
    name='Student Performance Prediction',
    version="0.0.1",
    author="Prince Choudhary",
    author_email="choudharyprince890@gmail.com",
    install_requirements = get_requirements("requirements.txt"),
    packages=find_packages()
)


