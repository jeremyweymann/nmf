'use strict';

const {Matrix} = require('ml-matrix');
const {gradientAdditive} = require('./algorithms/gradientAdditive');

/* module.exports = {
    nmf: nmf
}; */

module.exports = nmf;

/**
 * Compute the NMF of a matrix V, i.e the matrix W and H => A ~= W.H
 * @param {Matrix} V - Matrix to factorize
 * @param {Object} options - options can include the parameters k (width of the Matrix W and height of the Matrix H), the algorithm to perform, Winit (Init matrix of W), Hinit (Init matrix of H), tol (tolerance - default is 0.001) and maxIter (maximum of iterations before stopping - default is 100)
 * @return {Object} WH - object with the format {W: ..., H: ...}. W and H are the results (i.e A ~= W.H)
 */

function nmf(V, k, algorithm, options = {}) {


    let result = algorithm(V, k, options);
    let W = result.W;
    let H = result.H;

    /* optional value returned (?): 
    - Residual matrix i.e. V - W*H 
    - Sum of squares (only if Frobenius norm chosen?) i.e. ||V - W * H ||^2 = sum of ((V)ij - (W * H)ij)^2 
    - Generalised Kullback-Leibler divergence (only KL div choosen?) i.e. D(V||W * H) = sum of (V)ij . log((V)ij / (W * H)ij) - (V)ij + (W * H)ij
    - Sparsity of the resulting matrices 
    */ 

    return {W: W, H: H};
}

