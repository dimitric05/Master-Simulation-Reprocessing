# Master-Simulation-Reprocessing
This tool reprocesses production images to detect quality markings (commonly called NG marks) that may have been incorrectly applied during earlier inspection. 

Using OpenCV for image analysis, the program scans each image for these marks with better accuracy and faster processing using TurboJPEG. 

A simple Tkinter-based user interface allows you to load folders of images, run batch processing, and visually review results making it easier to validate or correct past inspection decisions.

Depending on the users specs, can process images up to ~650 images/sec.
