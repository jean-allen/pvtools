# PVTools: Tools for working with pressure-volume curves in Python.

This is a quick tool I whipped up perform basic analysis of P-V curves in python, since I couldn't find anything else out there. It's based around an object class called a PVCurve which, when instantiated, performs a bunch of basic calculations (e.g., saturated water content, turgor loss point, R²...). It also includes an automated breakpoint selection, so you can figure out where the transition between the two portions of the drydown occurs based on the possible R² values. There are also a couple of automated data visualizations.

For an example of the basic functionality, see test_data.py. Cheers!

Installation:  conda install jeanallen::pvtools 