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

template <int rows, int columns, int inners>
inline void matmulImplNaiveRegisterAcc(float *left, float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        acc +=
            left[row * columns + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
} } }




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

    auto start = chrono::high_resolution_clock::now();
    matmulImplNaive<dim, dim, dim>(left, right, result);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for matrix multiplication naive: " << (double)duration.count() << " millisecs" << endl;
    cout << "------------------------" << endl;


    start = chrono::high_resolution_clock::now();
    matmulImplNaiveRegisterAcc<dim, dim, dim>(left, right, result);
    end = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Time taken for matrix multiplication Register Acc: " << (double)duration.count() << " millisecs" << endl;
    cout << "------------------------" << endl;


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