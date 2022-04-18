## reflections
compile with `nvcc reflections.cu -lX11` as the program uses X11 for recognizing keyboard presses. 
With a good GPU the bottleneck in performance will be printing to the terminal, thus i recommend using a fast one (e.g. Alacritty).
