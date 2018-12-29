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
        if(Precision.equalsWithRelativeTolerance(xStart, xEnd, 0.001)) {
            return null
        }
        val slope = (yEnd - yStart) / (xStart - xEnd)
        if (FastMath.abs(slope) < MIN_ALLOWED_LANE_SLOPE) return null

        return slope
    }
}