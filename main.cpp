#include <iostream>
#include "MLP.h"
#include "Matrix.h"
#include <ctime>
#include <cstdlib>
#include <vector>

using std::vector;

int main(){
    srand(time_t(time(NULL)));

    MLP<2, 3, 2> mlp;

    vector<Matrix<2, 1>> samplesA(100), samplesB(100);
    for(int i = 0; i < 100; i++){
        samplesA[i](0, 0) = (float)rand()/INT32_MAX + 5;
        samplesA[i](1, 0) = (float)rand()/INT32_MAX + 5;
        samplesB[i](0, 0) = (float)rand()/INT32_MAX - 5;
        samplesB[i](1, 0) = (float)rand()/INT32_MAX - 5;
    }

    Matrix<2, 1> a, b;
    a(0, 0) = 1; a(1, 0) = 0;
    b(0, 0) = 0; b(1, 0) = 1;
    for(int epochIdx = 0; epochIdx < 100; epochIdx++){
        for(int sampleIdx = 0; sampleIdx < 100; sampleIdx++){
            mlp.train(samplesA[sampleIdx], a);
            mlp.train(samplesB[sampleIdx], b);
        } 
    }
    
    int falseA = 0; // just a test
    int falseB = 0;
    for(int i = 0; i < 100; i++){
        //std::cout << mlp(samplesA[i]) << std::endl; // just for debugging purposes!
        if(mlp(samplesA[i])(0, 0) <= mlp(samplesA[i])(1, 0)) falseA++;
        if(mlp(samplesB[i])(0, 0) >= mlp(samplesB[i])(1, 0)) falseB++;       
    }

    std::cout << "FalseA: " << falseA << std::endl;
    std::cout << "FalseB: " << falseB << std::endl;
}