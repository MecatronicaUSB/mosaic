# Automated Video Mosaic

Main repo for 2D mosaic generation pipeline.
Currently under development as part of undergrad project of Víctor García

**mosaic** is a free and open source software licensed under the [GNU GPLv3.0 License](https://en.wikipedia.org/wiki/GNU_General_Public_License), unless otherwise specified in particular modules or libraries (see LICENSE and README.md).

## Table of Contents
- [Modules list](#modules-list)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Software details](#software-details)
- [Contributing](#contributing)
- [License](#license)

## Modules list
- [core](https://github.com/MecatronicaUSB/mosaic/tree/master/core) Main project for automated mosaic build.
- [fcomp](https://github.com/MecatronicaUSB/mosaic/tree/master/modules/fcomp) Feature extractors and matchers comparison (extractors: SIFT, SURF, ORB, KAZE. matchers: Brute force, Flann)
- [stitch](https://github.com/MecatronicaUSB/mosaic/tree/master/modules/stitch) Image warping and stitching

Each module contains information describing its usage, with (usually) some useful README file. Code documentation is expected to be provided in Doxygen-compatible format. The current release contains C/C++ implementation.

## Directory Layout

    ├──core                # Core module
     \ ├── src             # source files
          └── ...
     \ ├── include         # header files
          └── ...  
     \ main.cpp
     \ ...
    ├── modules
     \ ├── fcomp           # module for festure extractor comparison
          ├── src          # source files
          ├── include      # header files
          └── ...          
     \ ├── stitch          # module for image stitching
          ├── src          # source files
          ├── include      # header files
          └── ...

## Requirements

The current release has been developed and tested in Ubuntu 16.04 LTS 64 bits

- [OpenCV 3.2+](http://opencv.org) and extra modules (OpenCV contrib).
- [cmake 2.8](https://cmake.org/) - cmake making it happen again

## Getting Started

To start using this project, proceed to the standard *clone* procedure:

```bash
cd <some_directory>
git clone https://github.com/MecatronicaUSB/mosaic.git
```

## Software Details

- Implementation done in C++.

## Contributing

Summary of contributing guidelines (based on those of OpenCV project):

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Use small datasets for testing purposes;
* Follow always the same coding style guide. If possible, apply code formating with any IDE.

## License

Copyright (c) 2017-2018 Grupo de Investigación y Desarrollo en Mecatrónica (<mecatronica@usb.ve>).
Released under the [GNU GPLv3.0 License](LICENSE). 
