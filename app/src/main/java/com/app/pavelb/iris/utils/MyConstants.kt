package com.app.pavelb.iris.utils

import org.opencv.core.MatOfPoint
import org.opencv.core.Point

const val IMAGE_WIDTH = 1280
const val IMAGE_HEIGHT = 720
const val IMAGE_MIDPOINT = IMAGE_WIDTH / 2
const val NUMBER_OF_FRAMES_BUFFER = 10
val ROI_POLYGON = MatOfPoint(Point(135.0, 720.0), Point(582.0, 457.0), Point(701.0, 457.0), Point(1145.0, 720.0))
const val MIN_ALLOWED_LANE_SLOPE = 0.2