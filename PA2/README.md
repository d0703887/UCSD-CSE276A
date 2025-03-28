## CSE 276A PA2
The objective of this exercise is to use the on-board camera to drive throught 4 way points: (0, 0, 0) -> (1, 0, 0) -> (1, 2, $\pi$) -> (0, 0, 0). 
Using the feedback from the detected landmarks to improve localization performance and drive more effectively. **More details can be found in the report.**


## Files
### mpi_control.py
Close-loop controller function which takes apriltag detection as feedback and then adjusts robot's trajectory.

### utils.py
Apriltag detection function.

## Demo
https://youtu.be/N5bIZA7h7wo?si=oz--Psqucw02C-VZ
