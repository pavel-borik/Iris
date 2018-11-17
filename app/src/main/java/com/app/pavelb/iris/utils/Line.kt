package com.app.pavelb.iris.utils

import org.apache.commons.math3.util.FastMath
import org.apache.commons.math3.util.Precision

class Line(
    val xStart: Double,
    val yStart: Double,
    val xEnd: Double,
    val yEnd: Double
) {
    fun getSlope(): Double? {
        if(Precision.equalsWithRelativeTolerance(yStart, yEnd, 0.001)) {
            return null
        }
        val slope = (xEnd - xStart) / (yStart - yEnd)
        if (FastMath.abs(slope) < 0.1) return null

        return slope
    }
}