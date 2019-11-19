'''Setup script for icnn.'''


from setuptools import setup


VERSION = '1.2.2'

if __name__ == '__main__':

    # Long description
    with open('./README.md') as f:
        long_description = f.read()

    # Setup
    setup(name='icnn',
          version=VERSION,
          description='Deep image reconstruction from CNN features (Inverting CNN)',
          long_description=long_description,
          long_description_content_type='text/markdown',
          author='Shen Guo-Hua, Kei Majima',
          author_email='brainliner-admin@atr.jp',
          maintainer='Shuntaro C. Aoki',
          maintainer_email='brainliner-admin@atr.jp',
          url='https://github.com/KamitaniLab/icnn',
          license='MIT',
          keywords='neuroscience, brain decoding, deep learning, image reconstruction',
          packages=['icnn'],
          install_requires=['numpy', 'scipy'])
