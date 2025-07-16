// ECG Classifier - Based on real MIT-BIH training data
// Training Accuracy: 0.9910
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "weights.h"  // Include trained weights

#define INPUT_DIM 46
#define HIDDEN1_DIM 128
#define HIDDEN2_DIM 64
#define HIDDEN3_DIM 32
#define OUTPUT_DIM 6

typedef ap_fixed<16, 8> fixed_t;
typedef ap_fixed<32, 16> acc_t;

// ReLU activation function
fixed_t relu(fixed_t x) {
    return (x > fixed_t(0)) ? x : fixed_t(0);
}

// Main classification function
void ecg_classify_trained(
    fixed_t features[INPUT_DIM], 
    fixed_t probabilities[OUTPUT_DIM], 
    int* predicted_class
) {
    #pragma HLS INTERFACE m_axi port=features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=probabilities bundle=gmem1
    #pragma HLS INTERFACE m_axi port=predicted_class bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return
    
    // First layer (46 -> 128)
    fixed_t hidden1[HIDDEN1_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete
    
    for(int i = 0; i < HIDDEN1_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_biases[i]);  // Add bias
        
        for(int j = 0; j < INPUT_DIM; j++) {
            #pragma HLS PIPELINE
            sum += features[j] * fixed_t(dense_weights[j * HIDDEN1_DIM + i]);  // Correct weight access
        }
        
        hidden1[i] = relu(sum);
    }
    
    // Second layer (128 -> 64)
    fixed_t hidden2[HIDDEN2_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete
    
    for(int i = 0; i < HIDDEN2_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_1_biases[i]);  // Add bias
        
        for(int j = 0; j < HIDDEN1_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden1[j] * fixed_t(dense_1_weights[j * HIDDEN2_DIM + i]);  // Correct weight access
        }
        
        hidden2[i] = relu(sum);
    }
    
    // Third layer (64 -> 32)
    fixed_t hidden3[HIDDEN3_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden3 complete
    
    for(int i = 0; i < HIDDEN3_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_2_biases[i]);  // Add bias
        
        for(int j = 0; j < HIDDEN2_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden2[j] * fixed_t(dense_2_weights[j * HIDDEN3_DIM + i]);  // Correct weight access
        }
        
        hidden3[i] = relu(sum);
    }
    
    // Output layer (32 -> 6) - Simplified version without softmax
    fixed_t output[OUTPUT_DIM];
    
    for(int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS UNROLL
        acc_t sum = fixed_t(dense_3_biases[i]);  // Add bias
        
        for(int j = 0; j < HIDDEN3_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden3[j] * fixed_t(dense_3_weights[j * OUTPUT_DIM + i]);  // Correct weight access
        }
        
        output[i] = fixed_t(sum);
        probabilities[i] = output[i];  // Use raw output as "probabilities"
    }
    
    // Find maximum value class
    fixed_t max_val = probabilities[0];
    *predicted_class = 0;
    
    for(int i = 1; i < OUTPUT_DIM; i++) {
        if(probabilities[i] > max_val) {
            max_val = probabilities[i];
            *predicted_class = i;
        }
    }
}
