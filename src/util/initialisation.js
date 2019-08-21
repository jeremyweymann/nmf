'use strict';

const {Matrix, SingularValueDecomposition} = require('ml-matrix');

module.exports = initialisation;

/**
 * Initialize the parameter for NMF
 * @param {Matrix} V 
 * @param {number} k 
 * @param {number} m 
 * @param {number} n 
 * @param {boolean} svdInitialisation 
 */

function initialisation(V, k, m, n, svdInitialisation) {
    let initParam;

    if (svdInitialisation === true) {
        initParam = initSVD(V, {k: k});
    } else {
        let Winit = Matrix.rand(m, k);
        let Hinit = Matrix.rand(k, n);
        initParam = {Winit, Hinit};
    }

    return initParam;
}

/**
 * Choose appropriate rank for nmf and good starting matrix (corresponding to given rank) 
 * @param {Matrix} V 
 * @param {object} options 
 * @param {number} [options.k]
 */

function initSVD(V, options = {}) {
    const {k = 0} = options;
    let svdV = new SingularValueDecomposition(V);
    let rank;
    if (k === 0) {
        let sumSingular = arrSum(svdV.s);
        let i = 0;
        let normalizedSum = 0;
        while (i < svdV.s.length && normalizedSum < 0.9) {
            normalizedSum += svdV.s[i] / sumSingular;
            i++;
        }
        rank = i;
    } else {
        rank = k;
    }
    let s = new Array(rank);
    for (let j = 0; j < rank; j++) {
        s[j] = Math.sqrt(svdV.s[j]);
    }

    let sqrtRightS = Matrix.diag(s, rank, V.columns);
    let sqrtLeftS = Matrix.diag(s, V.rows, rank);

    let W0 = svdV.U.subMatrix(0, V.rows - 1, 0, rank - 1).mmul(sqrtLeftS);
    let H0 = sqrtRightS.mmul(svdV.V.transpose());

    let Winit = W0.abs();
    let Hinit = H0.abs();

    return {Winit, Hinit, rank};
}


function arrSum(arr) {
    return arr.reduce(function (a, b) {
        return a + b;
    }, 0);
}
