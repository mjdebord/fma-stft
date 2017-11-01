# FMA-STFT (WIP)

This contains ipynb(s) that have some minimal examples on how to use Free Music Archive data.

Check out these references for more information and other more complicated examples:
[paper]:     https://arxiv.org/abs/1612.01840
[FMA]:       https://freemusicarchive.org
[github]:    https://github.com/mdeff/fma

## Contents
This repo currently has a ipynb that shows how to:
* Retrieve genre-specific songs from the fma_metadata files
* Compute STFTs on songs using scipy
* Display STFT information
* Invert STFTs back into waveforms
* Export waveforms back into .mp3 files
* Create Mel-scale spectrograms and invert those back into waveforms (wip)

This information is a bit redundant given the official fma github page, but it offers
simplified/alternative examples that could be useful for learning.

The heavy lifting was already done by authors of the FMA paper. Check out 
the listed github page for many more FMA utilities and links to larger FMA datasets.

The included utils.py in the fma subtree is from the listed github source at tags/rc1

## Usage
This repo has been tested on Ubuntu 16.04 with python 3.6

1. Download FMA 'small' audio subset (~7.2GiB) and the meta data (~342MiB). Check integrity and unzip.
	```sh
	curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
	curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
	echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
	echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
	unzip fma_metadata.zip
	unzip fma_small.zip
	```

2. I recommend using python >= 3.5 and a pyenv environment
	```sh
	pyenv install 3.6.0
	pyenv virtualenv 3.6.0 fmaenv
	pyenv activate fmaenv
	```

3. Clone the repository.
	```sh
	git clone https://github.com/mjdebord/fma-stft.git
	cd fma-stft
	```

4. Install the Python dependencies from `requirements.txt`. You may also need to install [ffmpeg] or [graphviz].
	```sh
	pip3 install -r requirements.txt
	```

5. This project makes use of dotenv. Make a .env file at the repo root directory
	```sh
	cat .env
	AUDIO_DIR=/path/to/unzipped/audio
	METADATA_DIR=/path/to/unzipped/metadata
	```

6. Open Jupyter and run fma-stft.ipynb
	```sh
	jupyter notebook
	```

[pyenv]:      https://github.com/pyenv/pyenv
[pyenv-virt]: https://github.com/pyenv/pyenv-virtualenv
[ffmpeg]:     https://ffmpeg.org/download.html
[graphviz]:   http://www.graphviz.org/

## FMA License & co
This is directly from https://github.com/mdeff/fma
Please feel free to use any of my additional code for any non-sinister purpose

* Please cite our [paper] if you use our code or data.
* The code in this repository is released under the terms of the
  [MIT license](LICENSE.txt).
* The metadata is released under the terms of the
  [Creative Commons Attribution 4.0 International License (CC BY 4.0)][ccby40].
* We do not hold the copyright on the audio and distribute it under the terms
  of the license chosen by the artist.
* The dataset is meant for research purposes.
* We are grateful to SWITCH and EPFL for hosting the dataset within the context
  of the [SCALE-UP] project, funded in part by the swissuniversities [SUC P-2
  program].

[ccby40]: https://creativecommons.org/licenses/by/4.0
[SCALE-UP]: https://projects.switch.ch/scale-up/
[SUC P-2 program]: https://www.swissuniversities.ch/isci
