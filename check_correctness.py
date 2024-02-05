import os
import re
import subprocess
from pathlib import Path

# Array of C++ files
files = ["sequential.cpp", "pthread_pipe.cpp"]

# Array of matrix configurations
mat_ns = [64, 256, 1024, 2048]

# Array of thread counts
threads = [2, 4, 8, 16]


stats = {}

# Loop over each file
for file in files:
    print("Compiling and running " + file)

    filename = Path(file).stem

    # Loop over each matrix configuration
    for mat_n in mat_ns:

        if file == "sequential.cpp":
           # Compile the file with the current matrix configuration
            subprocess.run(["g++", "-O3", "-std=c++11", "-o", "program", "-DMAT_N=" + str(mat_n), file])

            with open(f"out/{filename}_{mat_n}.txt", 'w') as f:
                subprocess.run(["./program"], stdout=f)
            # Run the program and store the output
            # subprocess.run(["./program", ">", f"{filename}_{mat_n}.txt"])

        else:
            # Loop over each thread count
            for thread in threads:
                
                if "mpi" in filename:
                    # Compile the file with the current matrix configuration and thread count
                    subprocess.run(["mpic++", "-O3", "-std=c++11", "-o", "program", "-DMAT_N=" + str(mat_n), "-DTHREADS=" + str(thread), file])
            
                    with open(f"out/{filename}_{mat_n}_{thread}.txt", 'w') as f:
                        subprocess.run(["mpirun", "-np", str(thread), "./program"], stdout=f)
            
                else:
                    # Compile the file with the current matrix configuration and thread count
                    subprocess.run(["g++", "-O3", "-std=c++11", "-lpthread", "-o", "program", "-DMAT_N=" + str(mat_n), "-DTHREADS=" + str(thread), file])

                    # Run the program and store the output
                    # subprocess.run(["./program", ">", f"{filename}_{mat_n}_{thread}.txt"])
                    with open(f"out/{filename}_{mat_n}_{thread}.txt", 'w') as f:
                        subprocess.run(["./program"], stdout=f)
            