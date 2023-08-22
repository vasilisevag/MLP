#pragma once

#include <cstdint>
#include "Matrix.h"

template <uint32_t R, uint32_t C>
Matrix<R, C - 1> RemoveBias(const Matrix<R, C>& matrix){
    Matrix<R, C - 1> resultMatrix;

    for(int r = 0; r < matrix.r(); r++)
        for(int c = 0; c < matrix.c() - 1; c++)
            resultMatrix(r, c) = matrix(r, c);

    return resultMatrix; 
}

template <uint32_t R>
Matrix<R + 1, 1> AddBias(const Matrix<R, 1>& matrix){
    Matrix<R + 1, 1> resultMatrix;

    for(int r = 0; r < matrix.r(); r++)
        resultMatrix(r, 0) = matrix(r, 0);
    resultMatrix(R, 0) = 1;

    return resultMatrix; 
}

template <uint32_t N, uint32_t L, uint32_t R, uint32_t... TS>
struct nth_type : nth_type<N - 1, R, TS...> {};

template <uint32_t L, uint32_t R, uint32_t... TS>
struct nth_type<0, L, R, TS...> {
    using value_type = Matrix<R, L+1>;
};

template <uint32_t L, uint32_t R, uint32_t... TS>
class Weights {
    public:
        Matrix<R, L + 1> weights;
        Weights<R, TS...> rest;
};

template <uint32_t L, uint32_t R>
class Weights<L, R> {
    public:
        Matrix<R, L + 1> weights;
};