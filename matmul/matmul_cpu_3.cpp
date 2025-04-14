// vectorized FMA instructions VFMADD - intel mkl
// 256 bit long YMM register --> YMM3 = (YMM1 * YMM2) + YMM3
// Single precision =  * 32 bit floats

// VFMADD -> 8 Mul + 8 Add
// VFMADD throughput = 0.5 cycles --> 32 FLOPs/ cycle

#include<iostream>
#include<chrono>

using namespace std;


template <int rows, int columns, int inners>
inline void matmulImplNaive(float *left, float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
      }
} } }



template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right,
                             float *result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
} } } } }



void printMatrix(float *matrix, int rows, int columns) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      cout << matrix[i * columns + j] << " ";
    }
    cout << endl;
  }
}



int main(){
    constexpr int dim = 1024;
    float *left = new float[dim * dim];
    float *right = new float[dim * dim];
    float *result = new float[dim * dim];

    // Initialize matrices
    for (int i = 0; i < dim * dim; i++) {
        left[i] = i;
        right[i] = i;
        result[i] = 0;
    }

    // printMatrix(left, dim, dim);
    // cout << "------------------------" << endl;
    // printMatrix(right, dim, dim);
    // cout << "------------------------" << endl;

    // Call the matrix multiplication function


    #ifdef NAIVE
    auto start = chrono::high_resolution_clock::now();
    matmulImplNaive<dim, dim, dim>(left, right, result);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for matrix multiplication naive: " << (double)duration.count() << " millisecs" << endl;
    cout << "------------------------" << endl;
    #endif

    #ifdef TILING
    auto start = chrono::high_resolution_clock::now();
    matmulImplTiling<dim, dim, dim, 32>(left, right, result);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for matrix multiplication with Tiling: " << (double)duration.count() << " millisecs" << endl;
    cout << "------------------------" << endl;
    #endif

    // Print the result
    //cout << "Result matrix:" << endl;
    //printMatrix(result, dim, dim);
    // Free allocated memory
    delete[] left;
    delete[] right;
    delete[] result;

    return 0;

}




// Time taken 
//Naive = 3.584 secs (g++ matmul_cpu.cpp -o matmul_cpu)
// with -O3 = 615 millisecs
// with -march=native = 315 millisecs
// with -ffast-math = 315 millisecs
// With Register Acc = 300 ms



// Time taken 
//Naive = 3.584 secs (g++ matmul_cpu.cpp -o matmul_cpu)
// with -O3 = 615 millisecs
// with -march=native = 315 millisecs
// with -ffast-math = 315 millisecs
// With Register Acc = 300 ms
// With Loop Reorder = 75 ms

// Cache miss with Register Acc: 847,054
// Cache miss with Loop Reorder: 756,187


/*


Time taken for matrix multiplication naive: 306 millisecs
------------------------

 Performance counter stats for './matmulcpu':

        10,009,199      cpu_atom/cache-misses/           #   67.95% of all cache refs           (0.26%)
           886,850      cpu_core/cache-misses/           #    0.33% of all cache refs           (99.74%)
        14,730,440      cpu_atom/cache-references/                                              (0.26%)
       268,739,531      cpu_core/cache-references/                                              (99.74%)

       0.315432183 seconds time elapsed

       0.308457000 seconds user
       0.006987000 seconds sys
s


Time taken for matrix multiplication Loop Reorder: 67 millisecs
------------------------

 Performance counter stats for './matmulcpu':

     <not counted>      cpu_atom/cache-misses/                                                  (0.00%)
           736,146      cpu_core/cache-misses/           #    1.09% of all cache refs         
     <not counted>      cpu_atom/cache-references/                                              (0.00%)
        67,714,755      cpu_core/cache-references/                                            

       0.078039728 seconds time elapsed

       0.070766000 seconds user
       0.007076000 seconds sys

*/
