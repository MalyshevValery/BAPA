### Floyd-Warshall algorithm with use of CUDA

*Prerequisites:*
CUDA
CMake >= 3.14

Note: compilation for CUDA require gcc-6. It can be achieved by symlinking the
gcc-6 to /usr/local/cuda/bin/gcc or wherever is your CUDA bin folder
```
    sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
```

**Build**
```
   
   cmake CMakeLists.txt
   make

```

Programs included in this project:

1. **CUDA\_FW** - Floyd-Warshall algorithm realization on CUDA
    ```
        CUDA_FW <input file> <output file> <algorithm type> <block size>
    ```
    Algorithm type can be either block_2d or block_3d

2. **FW** - Floyd-Warshall algorithm with use of std::thread
    ```
        FW <input file> <output file> <algorithm type> <block size>
    ```
    In addition to CUDA\_FW types has also sequential type (seq)

3. **GraphGen** - utility for random oriented graph generation
    ```
        GraphGen <number of vertices> <load (0,1)> <output file>
    ```
    Load is the number of generated edges represented by fracture of number of
    possible edges.
