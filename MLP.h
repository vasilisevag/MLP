#pragma once

#include "Weights.h"
#include "Matrix.h"
#include <cstdint>

template <uint32_t I, uint32_t L, uint32_t... TS> // we can generalize this one
constexpr uint32_t nth_layer_dimension(){
    if constexpr (I == 0)
        return L;
    else 
        return nth_layer_dimension<I - 1, TS...>();
}

template <typename IN, uint32_t... LAYERS>
Matrix<nth_layer_dimension<sizeof...(LAYERS) - 1, LAYERS...>(), 1> ForwardPropagation(const IN& in, const Weights<LAYERS...>& weights){
    auto inBiased = AddBias(in);
    
    if constexpr (sizeof...(LAYERS) - 2 == 0) 
        return ReLU(weights.weights * inBiased);
    else 
        return ForwardPropagation(ReLU(weights.weights * inBiased), weights.rest); 
}

template <uint32_t... LAYERS>
Matrix<nth_layer_dimension<0, LAYERS...>(), 1> BackPropagation(const Matrix<nth_layer_dimension<0, LAYERS...>(), 1>& in, const Matrix<nth_layer_dimension<sizeof...(LAYERS) - 1, LAYERS...>(), 1>& y, Weights<LAYERS...>& weights){
    auto inBiased = AddBias(in);
    auto z = weights.weights * inBiased;
    auto a = ReLU(z);

    Matrix<z.r(), 1> delta, deltaPart;
    if constexpr (sizeof...(LAYERS) - 2 == 0){ // this is under construction!
        delta = PointwiseMultiplication(a - y, ReLUDerivative(z));
    }
    else {
        deltaPart = BackPropagation(a, y, weights.rest);
        delta = PointwiseMultiplication(deltaPart, ReLUDerivative(z));              
    }
    
    auto deltaPreviousPart = T(RemoveBias(weights.weights)) * delta;
    weights.weights = weights.weights - 0.003 * delta * T(inBiased); // 0.03 is the learning rate

    return deltaPreviousPart;
}

template <uint32_t... LAYERS>
class MLP {
    public: 
        Matrix<nth_layer_dimension<sizeof...(LAYERS) - 1, LAYERS...>(), 1> evaluate(const Matrix<nth_layer_dimension<0, LAYERS...>(), 1>& in) {
            return ForwardPropagation(in, weights);
        }
        Matrix<nth_layer_dimension<sizeof...(LAYERS) - 1, LAYERS...>(), 1> operator()(const Matrix<nth_layer_dimension<0, LAYERS...>(), 1>& in) {
            return ForwardPropagation(in, weights);
        }
        void train(const Matrix<nth_layer_dimension<0, LAYERS...>(), 1>& a, const Matrix<nth_layer_dimension<sizeof...(LAYERS) - 1, LAYERS...>(), 1>& y){
            BackPropagation(a, y, weights);
        }
    //private: this is just for debugging purposes!
        Weights<LAYERS...> weights;
};