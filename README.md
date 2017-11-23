# TDSE_solver
Numerical TDSE Solver

Dependencies: Python 3.6, matplotlib/numpy

Optoinal Dependencies: ffmpeg for video encoding

If you don't have ffmpeg, comment the anim.save line in plotutil.py and replace it with plt.show() to view video.

Help: python tdse.py -h

Current issues with problem=barrier. Try changing spatial and time resolution (J, N). Space and time boundaries (minmaxx, minmaxt). barrier(x) height and width. (np.peicewise arguments).
