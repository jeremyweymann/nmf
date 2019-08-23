'use strict';

const {
    Matrix,
    WrapperMatrix2D,
    SingularValueDecomposition
} = require('ml-matrix');
const anls = require('./algorithms/anls');

const initialisation = require('./util/initialisation');

/* module.exports = {
    nmf: nmf
}; */

/**
 * @class nmf
 * @param {Matrix} V
 * @param {number} k
 * @param {object} [options={}]
 * @param {boolean} [svdInitialisation=false]
 * @param {function} [options.algorithm="anls"]
 * @param {function} [options.algorithmOptions={}]
 * @param {function} [options.algorithmOptions.method="fcnnls"]
 * @param {function} [options.algorithmOptions.maxIterations]
 * @param {Matrix} [options.Winit]
 * @param {Matrix} [options.Hinit]
 * @param {number} [options.tolerance]
 */

class nmf {
    constructor(V, k, options = {}) {
        const {svdInitialisation = false} = options;

        this.k = k;
        this.V = WrapperMatrix2D.checkMatrix(V);

        if (typeof k !== 'number') {
            throw new Error('k must be a number');
        }

        let {Winit, Hinit, rank} = initialisation(
            this.V,
            this.k,
            svdInitialisation
        );

        const {algorithm = 'anls', method = 'fcnnls', algorithmOptions} = options;

        let algorithmFct;
        switch (algorithm.toLowerCase()) {
            case 'anls':
                algorithmFct = anls;
                break;
            default:
                throw new Error('Undefined algorithm: ' + algorithm);
        }

        //let algOptions = {tol, maxIterations};

        let result = algorithmFct(V, k, Winit, Hinit, algorithmOptions);
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
