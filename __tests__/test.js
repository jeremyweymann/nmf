'use strict';

const nmf = require('../src');
const gradientAdditive = require('../src/algorithms/gradientAdditive');
const anlsActiveSet = require('../src/algorithms/anlsActiveSet');

const {Matrix} = require('ml-matrix');
const {toBeDeepCloseTo} = require('jest-matcher-deep-close-to');
expect.extend({toBeDeepCloseTo});

describe('NMF test', () => {
    it('use case 1', async () => {
        let W0 = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]);
        let H0 = new Matrix([[1, 2], [3, 4], [5, 6], [7, 8]]);
        let Winit = new Matrix([[1, 1, 1, 1], [1, 1, 1, 1]]);
        let Hinit = new Matrix([[1, 1], [1, 1], [1, 1], [1, 1]]);

        let V = W0.mmul(H0);

        const options = {
            Winit: Winit,
            Hinit: Hinit,
            tol: 0.0001,
            maxIter: 200
        };

        const nmfResult = nmf(V, 2, gradientAdditive, options);
        const W = nmfResult.W;
        const H = nmfResult.H;
        let result = W.mmul(H);

        console.log({alg: 'GradientDescent', V, result, W, H});
        expect(result.to2DArray()).toBeDeepCloseTo(V.to2DArray(), 1);
    });

    it.only('Test ANLS-FCNNLS', async () => {

        //let V = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        //let V = Matrix.randInt(100, 20);
        //let W0 = new Matrix([[1, 2], [3, 4], [5, 6], [7, 8]]);
        //let H0 = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]);
        //let Winit = new Matrix([[1, 1], [1, 1], [1, 1], [1, 1]]);
        //let Hinit = new Matrix([[1, 1, 1, 1], [1, 1, 1, 1]]);


        //let V = W0.mmul(H0);

        let V = Matrix.randInt(5, 4);

        const nmfResult = nmf(V, 2, anlsActiveSet, {maxIter: 20000, regularization: 'L'});
        let W = nmfResult.W;
        let H = nmfResult.H;
        let result = W.mmul(H);

        console.log({alg: 'ANLS-ActiveSet H (right) regularized', V, result, W, H});
        expect(result.to2DArray()).toBeDeepCloseTo(V.to2DArray(), 1);
    });
});
