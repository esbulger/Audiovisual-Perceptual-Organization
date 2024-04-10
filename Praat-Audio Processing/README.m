# Audio Stimuli Processing Guide

This repository contains guidelines and scripts for processing audio files for research studies. The process involves noise reduction, cutting, duration adjustment, and intensity normalization of audio files using Audacity and Praat software.

## Getting Started

### Prerequisites
- [Audacity](https://www.audacityteam.org/download/)
- [Praat](http://www.fon.hum.uva.nl/praat/)
- Praat Vocal Toolkit ([Download](http://www.praatvocaltoolkit.com/))

### Noise Reduction and Cutting in Audacity

1. **Load Files:** Open Audacity, navigate to `File > Open` and select your raw audio files.
2. **Reduce Noise:** Select a portion of background noise, use `Effect > Noise Reduction…`, get the noise profile, then apply noise reduction to the entire sound.
3. **Cut Audio:** Select and delete unwanted parts of the audio. Aim for a clean cut of just the desired sound.
4. **Export:** Use `File > Export > Export as WAV` to save the processed file. For multiple files, `Export Multiple…` can speed up the process.

### Duration Adjustment and Intensity Normalization in Praat

1. **Load Processed Files:** In Praat, go to `Open > Read from file…` and select your processed files from Audacity.
2. **Adjust Duration:** Select all files, then use `Process > Change duration` to set the new duration to 0.41756 seconds with the Stretch method.
3. **Normalize Intensity:** Apply "Loudness Normalization" to normalize the RMS level to -20 dB.

### Saving Files with Praat Script

A script is provided for saving multiple files in Praat. Modify the directory path in the script to match your destination folder. Run this script in Praat to save all processed sounds.

```praat
# Script for saving multiple sounds in Praat
form Enter directory to save your sounds
     sentence directory /path/to/your/destination/folder/
endform
# Script contents omitted for brevity
```

### Renaming Files

To remove the `changeduration_Stretch__0_41756` suffix from file names:
- Select all files in your file explorer.
- Right-click and choose "Rename".
- Replace `-changeduration_Stretch__0_41756` with nothing to clean up the file names.

