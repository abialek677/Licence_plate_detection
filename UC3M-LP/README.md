# UC3M-LP

UC3M-LP dataset is a comprehensive and open-source collection of annotated images for European (Spanish) license plate detection and recognition tasks. Researchers and developers are encouraged to use the UC3M-LP dataset to develop and evaluate algorithms for license plate detection, localization, character segmentation, and optical character recognition (OCR). The dataset also supports various research areas such as vehicle surveillance, automated toll systems, traffic analysis, and security applications. Check the [open-access paper](https://doi.org/10.1016/j.robot.2023.104608).

It is the largest open-source dataset for European license plate detection and recognition and the first one ever dedicated to Spanish license plates. It contains 1975 images from 2547 different vehicles with their corresponding license plate, comprising a total of 12757 plate characters.

<img src="docs/dataset.png" alt="UC3M-LP dataset"/>


## Overview

The dataset is split into train, with 1580 images (80%), and test, with 395 images (20%). Each one of the 2547 license plates has been labeled with a two letter code. The first letter refers to the typology of Spanish license plates:

- Type A: 2498 samples of the most common long, one row with white background.

<p align='center'>
    <img src="docs/lp_a.jpg" alt="Type A license plate" width=200/>
</p>

- Type B: 31 samples of motorcycle double row and white background.

<p align='center'>
    <img src="docs/lp_b.png" alt="Type B license plate" width=100/>
</p>

- Type C: 1 sample of light motorcycle one row with yellow background.

<p align='center'>
    <img src="docs/lp_c.png" alt="Type C license plate" width=100/>
</p>

- Type D: 11 samples of taxis and VTC (Spanish acronym for private hire vehicle) with blue background.

<p align='center'>
    <img src="docs/lp_d.png" alt="Type D license plate" width=200/>
</p>

- Type E: 6 samples of trailer tows with black characters and red background.

<p align='center'>
    <img src="docs/lp_e.jpg" alt="Type E license plate" width=200/>
</p>


and the second letter refers to the lighting conditions:

- Type D: 2185 samples at daytime.
- Type N: 362 samples at nighttime.


## Citation

If you use this dataset in your research, please cite the following paper:

```
@article{RamajoBallester2024,
    title = {Dual license plate recognition and visual features encoding for vehicle identification},
    journal = {Robotics and Autonomous Systems},
    volume = {172},
    pages = {104608},
    year = {2024},
    issn = {0921-8890},
    doi = {https://doi.org/10.1016/j.robot.2023.104608},
    url = {https://www.sciencedirect.com/science/article/pii/S0921889023002476},
    author = {Álvaro Ramajo-Ballester and José María {Armingol Moreno} and Arturo {de la Escalera Hueso}},
    keywords = {Deep learning, Public dataset, ALPR, License plate recognition, Vehicle re-identification, Object detection},
}
```

## Download

The dataset is available for [download](https://doi.org/10.21950/OS5W4Z).


## Transform to YOLO format

The dataset is provided in a custom format. To transform it to YOLO format, please follow these instructions:

1. Clone the repository:

```bash
git clone https://github.com/ramajoballester/UC3M-LP.git
cd UC3M-LP
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download and extract the dataset files and create this directory structure:

```
path/to/UC3M-LP/
└─── train
│   └─── 00001.jpg
│   └─── 00001.json
│   └─── 00002.jpg
│   └─── 00002.json
│   └─── ...
└─── test
│   └─── 00001.jpg
│   └─── 00001.json
│   └─── 00002.jpg
│   └─── 00002.json
│   └─── ...
└─── train.txt
└─── test.txt
```

4. Run the script to transform the dataset to YOLO format. It will create 2 versions of the dataset, one for LP detection from the whole image and another one for LP recognition from the cropped LP region. The script will resize the images to the specified dimensions and save the resulting images and labels in in new directories. You can specify the desired dimensions for the images as arguments of the script.

```bash
python scripts/labels2yolo.py path/to/UC3M-LP 320 160
```

Run `python scripts/labels2yolo.py -h` for more information.