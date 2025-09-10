
This is the repo for the paper **OT-PCA: New Key-Recovery Plaintext-Checking Oracle Based Side-Channel Attacks on HQC with Offline Templates**. (Link: https://eprint.iacr.org/2024/1715.pdf)

### Overview

The source code for finding patterns, creating templates, and simulations is located in the `src/` folder. 

### Folder Structure

```
main folder/
├── bin/
│   └── sh files for running the source code in src/
├── hqc128/
│   └── clean/
├── src/
│   ├── find_error_pattern.py  # Script to optimize error patterns
│   ├── find_group.py          # Script to find a group of error patterns
│   ├── create_template.py     # Script to generate templates
│   ├── simulattion.py         # Script for simulations
│   └── util.py                # Utility functions for the pipeline
└── config.json                # Configuration file for schemes  
```

### Note

The `hqc128/clean` directory contains the standard PQClean implementation of hqc-128 from [PQClean](https://github.com/PQClean/PQClean), with a single modification made by the paper's authors to facilitate their analysis.

A custom function, `reed_muller_decode_one_block`, was added to the `hqc128/clean/reed_muller.c` file. Its purpose is to retrieve the decoding output for a single Reed-Muller block, which is used for the offline error pattern finding and template construction of the FD attack.