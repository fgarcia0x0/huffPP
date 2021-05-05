# huffPP
The huffPP (Huffman Coding Plus Plus) is a utility tool written in C++ 20 to compact and unzip files using the huffman coding algorithm.

## System Requirements
- compiler that support at least C++ 20
- cmake
- ninja

## How to Compile?
**Enter in your terminal:**

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug|Release
```
**Go to the build folder, then:**
```sh
cmake --build . -- -j 4
```
