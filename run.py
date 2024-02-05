import os
import re
import subprocess
from pathlib import Path

# Array of C++ files
files = ["sequential.cpp", "mpi.cpp", "mpi_pipe.cpp", "pthread.cpp", "pthread_pipe.cpp"]

# Array of matrix configurations
mat_ns = [64, 256, 1024, 2048]

# Array of thread counts
threads = [2, 4, 8, 16]

# Output file
output_file = "./out/output.txt"

stats = {}
speedup = {}

# Loop over each file
for file in files:
    print("Compiling and running " + file)

    filename = Path(file).stem
    stats[filename] = {}
    speedup[filename] = {}

    # Loop over each matrix configuration
    for mat_n in mat_ns:
        stats[filename][mat_n] = {}
        speedup[filename][mat_n] = {}

        if file == "sequential.cpp":
           # Compile the file with the current matrix configuration
            subprocess.run(["g++", "-O3", "-std=c++11", "-o", "program", "-DMAT_N=" + str(mat_n), file])

            # Run the program and store the output
            output = subprocess.check_output(["./program"]).decode('utf-8')

            # Extract the time from the output
            time = re.search(r'\d+(\.\d+)?', output).group()

            stats[filename][mat_n][1] = time

        else:
            # Loop over each thread count
            for thread in threads:
                
                if "mpi" in filename:
                    # Compile the file with the current matrix configuration and thread count
                    subprocess.run(["mpic++", "-O3", "-std=c++11", "-o", "program", "-DMAT_N=" + str(mat_n), "-DTHREADS=" + str(thread), file])
            
                    # Run the program and store the output
                    output = subprocess.check_output(["mpirun", "-np", str(thread), "./program"]).decode('utf-8')
            
                    # Extract the time from the output
                    time = re.search(r'(Elapsed Time: )(\d+(\.\d+)?)', output).group(2)
            
                    stats[filename][mat_n][thread] = time
                    speedup[filename][mat_n][thread] = float(stats["sequential"][mat_n][1]) / float(time)
                
                else:
                    # Compile the file with the current matrix configuration and thread count
                    subprocess.run(["g++", "-O3", "-std=c++11", "-lpthread", "-o", "program", "-DMAT_N=" + str(mat_n), "-DTHREADS=" + str(thread), file])

                    # Run the program and store the output
                    output = subprocess.check_output(["./program"]).decode('utf-8')

                    # Extract the time from the output
                    time = re.search(r'\d+(\.\d+)?', output).group()

                    stats[filename][mat_n][thread] = time
                    speedup[filename][mat_n][thread] = float(stats["sequential"][mat_n][1]) / float(time)


content = ""

content += "{:<15s}{:<15s}".format("MAT_N", "Threads")
for thread_count in threads:
    content += "{:<15s}".format(str(thread_count))
content += "\n"
content += "\n"

for mat_n in mat_ns:
    
    for file in files:
        filename = Path(file).stem
        
        content += "{:<15s}".format(str(mat_n))
        content += "{:<15s}".format(filename)

        for thread_count in threads:
            if file != "sequential.cpp":  
                content += "{:<15s}".format(str(stats[filename][mat_n][thread_count]))
            else:
                content += "{:<15s}".format(str(stats[filename][mat_n][1]))
        content += "\n"
    
    content += "\n"

content += "\n"

content += "{:<15s}{:<15s}".format("MAT_N", "Threads")
for thread_count in threads:
    content += "{:<15s}".format(str(thread_count))
content += "\n"
content += "\n"

for mat_n in mat_ns:
    
    for file in files:
        filename = Path(file).stem

        if file != "sequential.cpp":
            content += "{:<15d}".format(mat_n)
            content += "{:<15s}".format(filename)            
            for thread_count in threads:
                content += "{:<15.2f}".format(speedup[filename][mat_n][thread_count])
            content += "\n"
        
    content += "\n"

# Write the table header to the output file
with open(output_file, 'w') as f:
    f.write(content)
    # f.write(str(stats))

