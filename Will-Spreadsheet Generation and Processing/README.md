# Gorilla Task Spreadsheet Generation Guide

This guide outlines the process for preparing and organizing stimuli into spreadsheets, facilitating efficient experiment design within the Gorilla platform. The process involves several steps, each corresponding to a specific Jupyter notebook file, to ensure the generated stimuli are properly randomized, and instruction and practice sections are appropriately set up for your experiments.

## Overview of Files and Their Purpose

1. **Combine_female_male_excel.ipynb**
   - **Purpose:** Randomly selects an even number of male and female trials to eliminate gender as a varying dimension. This step is crucial for reducing potential biases related to gender differences in the data analysis phase. Additionally, it reduces the total number of trials to streamline the experiment length.
   - **Key Actions:** 
     - Ensures gender balance across trials.
     - Reduces experiment length by limiting the number of trials.

2. **Task_generation.ipynb**
   - **Purpose:** Generates the spreadsheet for the formal Perception/Accuracy & Speed/Difficulty (P/A & S/D) tasks. This file is essential for structuring the core parts of the experiment, where participants' responses to stimuli are recorded and analyzed.
   - **Key Actions:** 
     - Produces the main task spreadsheet required for the experiment.

3. **Instruction_generation.ipynb** and **Instruction_modified.ipynb**
   - **Purpose:** These files are used to select, rename, and organize `.wav` files for the instruction section in Gorilla. They also generate the spreadsheets necessary for this section of the experiment. Instructions are a critical part of the experiment, ensuring participants understand the tasks they will perform.
   - **Key Actions:** 
     - Prepares instruction materials (audio files and spreadsheets) for the experiment.
   - **Note:** Use `Instruction_modified.ipynb` for updated or specific instruction requirements.

4. **Practice_generation.ipynb**
   - **Purpose:** Similar to the instruction files but focused on the practice section. This notebook selects, renames, and organizes `.wav` files for practice trials and generates corresponding spreadsheets. The practice section helps participants get accustomed to the experiment format without affecting the actual data collection.
   - **Key Actions:** 
     - Sets up the practice section with all necessary materials for a smooth participant onboarding.

## Workflow

1. **Begin with `Combine_female_male_excel.ipynb`:** Ensure that your stimuli are evenly distributed by gender and that the total trial count is optimized for your experiment's length.
2. **Proceed to `Task_generation.ipynb`:** Generate the main task spreadsheet, organizing your experiment's core stimuli and response metrics.
3. **Set up instructions with `Instruction_generation.ipynb` or `Instruction_modified.ipynb`:** Prepare the instructional materials tailored to your experiment's needs.
4. **Finalize with `Practice_generation.ipynb`:** Prepare practice trials to ensure participants are well-equipped to perform the actual tasks.

## Additional Notes

- The sample instructions and additional resources can be downloaded directly from the Gorilla Experiment platform (refer to V4-**-Instruction for more details).
- Ensure you have the Gorilla toolkit and necessary Python packages installed to run these notebooks effectively.

For more detailed instructions on each step, refer to the comments and documentation within each Jupyter notebook file. 
