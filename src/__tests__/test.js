'use strict';

const nmf = require('..');
const projGradDescent = require('../methods/methodsANLS/projGradDescent');

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

    it('Test ANLS-FCNNLS', async () => {

        //let V = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        //let V = Matrix.randInt(100, 20);
        //let W0 = new Matrix([[1, 2], [3, 4], [5, 6], [7, 8]]);
        //let H0 = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]);
        //let Winit = new Matrix([[1, 1], [1, 1], [1, 1], [1, 1]]);
        //let Hinit = new Matrix([[1, 1, 1, 1], [1, 1, 1, 1]]);


        //let V = W0.mmul(H0);

        //let V = new Matrix([[1, 2, 3, 4, 6], [2, 5, 23, 6, 90], [12, 4, 68, 0, 0], [5, 5, 5, 78, 100], [13, 32, 4756, 44, 4], [0, 1, 6, 9, 8], [123, 456, 789, 1013, 1]]);

        let V = Matrix.randInt(100, 20);

        const nmfResult = new nmf(V, 5, false);
        let W = nmfResult.W;
        let H = nmfResult.H;
        let result = W.mmul(H);

        //console.log({alg: 'ANLS-ActiveSet', V, result, W, H});
        expect(result.to2DArray()).toBeDeepCloseTo(V.to2DArray(), 1);
    });

    it.only('Test ANLS-PGDNNLS', async () => {

        //let V = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        //let V = Matrix.randInt(100, 20);
        //let W0 = new Matrix([[1, 2], [3, 4], [5, 6], [7, 8]]);
        //let H0 = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]);
        //let Winit = new Matrix([[1, 1], [1, 1], [1, 1], [1, 1]]);
        //let Hinit = new Matrix([[1, 1, 1, 1], [1, 1, 1, 1]]);


        //let V = W0.mmul(H0);

        //let V = new Matrix([[1, 2, 3, 4, 6], [2, 5, 23, 6, 90], [12, 4, 68, 0, 0], [5, 5, 5, 78, 100], [13, 32, 4756, 44, 4], [0, 1, 6, 9, 8], [123, 456, 789, 1013, 1]]);

        let V = Matrix.randInt(5, 3);

        const nmfResult = new nmf(V, 2, false, {method: projGradDescent});
        let W = nmfResult.W;
        let H = nmfResult.H;
        let result = W.mmul(H);

        console.log({alg: 'ANLS-projGradDescent', V, result, W, H});
        expect(result.to2DArray()).toBeDeepCloseTo(V.to2DArray(), 1);
    });
});
