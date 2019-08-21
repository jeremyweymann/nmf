'use strict';

const {Matrix, WrapperMatrix2D, SingularValueDecomposition} = require('ml-matrix');
const anls = require('./algorithms/anls');
const {fcnnls} = require('ml-fcnnls');
const initialisation = require('./util/initialisation');


/* module.exports = {
    nmf: nmf
}; */


/**
 * @class nmf
 * @param {Matrix} V
 * @param {number} k
 * @param {boolean} svdInitialisation
 * @param {object} [options={}]
 * @param {function} [options.algotithm]
 * @param {function} [options.method]
 * @param {Matrix} [options.Winit]
 * @param {Matrix} [options.Hinit]
 * @param {number} [options.tol] 
 * @param {number} [options.maxIterations]
 */

class nmf {
    constructor(V, k, svdInitialisation = false, options = {}) {
        V = WrapperMatrix2D.checkMatrix(V);
        var m = V.rows;
        var n = V.columns;

        this.k = k;
        this.V = V;
        this.m = m;
        this.n = n;

        if (typeof (k) !== 'number') {
            throw new Error('k must be a number');
        }

        let initParam = initialisation(V, k, m, n, svdInitialisation);

        const {algorithm = anls, method = fcnnls, Winit = initParam.Winit, Hinit = initParam.Hinit, tol, maxIterations} = options;

        if (typeof (algorithm) !== 'function') {
            throw new Error('algorithm must be a function');
        }

        //let algOptions = {tol, maxIterations};

        let result = algorithm(V, k, method, Winit, Hinit);
        this.W = result.W;
        this.H = result.H;

    }


}

/* optional value returned (?): 
    - Residual matrix i.e. V - W*H 
    - Sum of squares (only if Frobenius norm chosen?) i.e. ||V - W * H ||^2 = sum of ((V)ij - (W * H)ij)^2 
    - Generalised Kullback-Leibler divergence (only KL div choosen?) i.e. D(V||W * H) = sum of (V)ij . log((V)ij / (W * H)ij) - (V)ij + (W * H)ij
    - Sparsity of the resulting matrices 
    */ 

module.exports = nmf;

/**
 * Choose appropriate rank for nmf and good starting matrix (corresponding to given rank) 
 * @param {Matrix} V 
 * @param {object} options 
 * @param {number} [options.k]
 */

