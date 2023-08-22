#pragma once

#include <cstdint>
#include <iostream>
#include <ctime>

template <uint32_t R, uint32_t C, typename TYPE = double>
class Matrix {
    public:
        Matrix() {
            for(int r = 0; r < R; r++)
                for(int c = 0; c < C; c++)
                    values[r][c] = (TYPE)rand()/RAND_MAX;
        }
        constexpr uint32_t r() const {return R;}
        constexpr uint32_t c() const {return C;}
        TYPE& operator()(uint32_t r, uint32_t c) {
            return values[r][c];
        }
        const TYPE& operator()(uint32_t r, uint32_t c) const {
            return values[r][c];
        }
    private:
        double values[R][C];
};

template <uint32_t R, uint32_t C, uint32_t K>
Matrix<R, C> operator*(const Matrix<R, K>& lhs, const Matrix<K, C>& rhs) {
    Matrix<R, C> result;
    
    for(int r = 0; r < R; r++){
        for(int c = 0; c < C; c++){
            result(r, c) = 0;
            for(int k = 0; k < K; k++){
                result(r, c) += lhs(r, k) * rhs(k, c);
            }
        }
    }

    return result;
}

template <uint32_t R, uint32_t C>
Matrix<R, C> ReLU(const Matrix<R, C>& matrix){
    Matrix<R, C> resultMatrix;
    for(int r = 0; r < matrix.r(); r++)
        for(int c = 0; c < matrix.c(); c++)
            resultMatrix(r, c) = matrix(r, c) > 0 ? matrix(r, c) : 0;
    
    return resultMatrix;
}

template <uint32_t R, uint32_t C>
Matrix<R, C> PointwiseMultiplication(const Matrix<R, C>& lhs, const Matrix<R, C>& rhs){
    Matrix<R, C> resultMatrix;
    for(int r = 0; r < lhs.r(); r++)
        for(int c = 0; c < lhs.c(); c++)
            resultMatrix(r, c) = lhs(r, c) * rhs(r, c);

    return resultMatrix;
}

template <uint32_t R, uint32_t C>
Matrix<R, C> operator-(const Matrix<R, C>& lhs, const Matrix<R, C>& rhs){
    Matrix<R, C> resultMatrix;
    for(int r = 0; r < lhs.r(); r++)
        for(int c = 0; c < lhs.c(); c++)
            resultMatrix(r, c) = lhs(r, c) - rhs(r, c);

    return resultMatrix;
}

template <uint32_t R, uint32_t C>
Matrix<C, R> T(const Matrix<R, C>& matrix){
    Matrix<C, R> resultMatrix;

    for(int r = 0; r < matrix.r(); r++)
        for(int c = 0; c < matrix.c(); c++)
            resultMatrix(c, r) = matrix(r, c);

    return resultMatrix; 
}

template <uint32_t R, uint32_t C>
Matrix<R, C> ReLUDerivative(const Matrix<R, C>& matrix){
    Matrix<R, C> resultMatrix;
    for(int r = 0; r < matrix.r(); r++)
        for(int c = 0; c < matrix.c(); c++)
            resultMatrix(r, c) = matrix(r, c) > 0 ? 1 : 0; 

    return resultMatrix;
}

template <uint32_t R, uint32_t C, typename TYPE>
Matrix<R, C, TYPE> operator*(const TYPE& t, const Matrix<R, C, TYPE>& matrix){
    Matrix<R, C, TYPE> resultMatrix;
    for(int r = 0; r < matrix.r(); r++)
        for(int c = 0; c < matrix.c(); c++)
            resultMatrix(r, c) = matrix(r, c) * t;

    return resultMatrix;
}

template <uint32_t R, uint32_t C, typename TYPE>
Matrix<R, C, TYPE> operator*(const Matrix<R, C, TYPE>& matrix, const TYPE& t){
    return t * matrix;
}

template <uint32_t R, uint32_t C>
std::ostream& operator<<(std::ostream& out, const Matrix<R, C>& matrix){
    for(int r = 0; r < matrix.r(); r++){
        for(int c = 0; c < matrix.c(); c++)
            std::cout << matrix(r, c) << ' ';
        std::cout << std::endl;
    }
    return out;
}