package org.numjvm.vector;

import jdk.incubator.vector.FloatVector;

/** Implemented by one hidden class generated per supported graph. */
interface FloatExpression {
    FloatVector evaluate(FloatVector[] inputs);
}
