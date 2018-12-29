package com.app.pavelb.iris

import android.os.Bundle
import org.opencv.android.CameraBridgeViewBase
import org.opencv.imgproc.Imgproc
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import android.view.View.OnTouchListener
import android.app.Activity
import android.util.Log
import android.view.*
import com.app.pavelb.iris.utils.*
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.opencv.core.*
import org.opencv.utils.Converters
import java.util.*
import org.apache.commons.math3.fitting.WeightedObservedPoints
import org.apache.commons.math3.fitting.PolynomialCurveFitter
import org.opencv.core.Mat
import kotlin.math.max


class RecognitionActivity : Activity(), OnTouchListener, CvCameraViewListener2 {
    private val imageWidth: Int = IMAGE_WIDTH
    private val imageHeight: Int = IMAGE_HEIGHT
    private var previewSize: Size? = null
    private var roiPolygon: MatOfPoint? = null
    private var cameraFrameRgba: Mat? = null
    private var cameraFrameHls: Mat? = null
    private var cameraFrameHsv: Mat? = null
    private var cameraFrameCropped: Mat? = null
    private var cameraFrameGrayScale: Mat? = null
    private var frameMaskWhite: Mat? = null
    private var frameMaskYellow: Mat? = null
    private var frameMaskTotal: Mat? = null
    private var cameraFrameWhiteYellowMasked: Mat? = null
    private var perspTransformMat: Mat? = null
    private var srcPersp: Mat? = null
    private var dstPersp: Mat? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    private var allLines = LinkedList<Line>()
    private val leftLines = LinkedList<Line>()
    private val rightLines = LinkedList<Line>()

    private val regression = SimpleRegression(true)
    private val polyFitter = PolynomialCurveFitter.create(2)
    private val obs = WeightedObservedPoints()
    private val leftCoefs: Deque<Pair<Double, Double>> = ArrayDeque(NUMBER_OF_FRAMES_BUFFER)
    private val rightCoefs: Deque<Pair<Double, Double>> = ArrayDeque(NUMBER_OF_FRAMES_BUFFER)

    private var cameraMode = 5
    private var extrapolationMode = 5

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView!!.enableView()
                    mOpenCvCameraView!!.setOnTouchListener(this@RecognitionActivity)
                    mOpenCvCameraView!!.setMaxFrameSize(imageWidth, imageHeight)
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
    }

    /** Called when the activity is first created.  */
    public override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_recognition)

        mOpenCvCameraView = findViewById(R.id.cameraSurfaceView)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE

        mOpenCvCameraView!!.systemUiVisibility =
                View.SYSTEM_UI_FLAG_LOW_PROFILE or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION

        mOpenCvCameraView!!.setCvCameraViewListener(this)
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        previewSize = Size(imageWidth.toDouble(), imageHeight.toDouble())
        cameraFrameRgba = Mat(height, width, CvType.CV_8UC3)
        cameraFrameHls = Mat(height, width, CvType.CV_8UC3)
        cameraFrameHsv = Mat(height, width, CvType.CV_8UC3)
        cameraFrameCropped = Mat(height, width, CvType.CV_8UC3)
        cameraFrameGrayScale = Mat(height, width, CvType.CV_8UC3)
        frameMaskWhite = Mat(height, width, CvType.CV_8UC1)
        frameMaskYellow = Mat(height, width, CvType.CV_8UC1)
        frameMaskTotal = Mat(height, width, CvType.CV_8UC1)
        cameraFrameWhiteYellowMasked = Mat(height, width, CvType.CV_8UC3)
        perspTransformMat = Mat()
        roiPolygon = ROI_POLYGON

        srcPersp = Converters.vector_Point2f_to_Mat(
            listOf(
                Point(190.0, 720.0),
                Point(582.0, 457.0),
                Point(701.0, 457.0),
                Point(1145.0, 720.0)
            )
        )
        dstPersp = Converters.vector_Point2f_to_Mat(
            listOf(
                Point(200.0, 720.0),
                Point(200.0, 0.0),
                Point(1000.0, 0.0),
                Point(1000.0, 720.0)
            )
        )
    }

    override fun onCameraViewStopped() {
        cameraFrameRgba!!.release()
    }

    override fun onTouch(v: View, event: MotionEvent): Boolean {
        val x = event.x.toInt()
        if (x < IMAGE_MIDPOINT) {
            cameraMode = ++cameraMode % 6
        } else {
            extrapolationMode = ++extrapolationMode % 6
        }


        return false // don't need subsequent touch events
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat? {
        cameraFrameRgba = inputFrame.rgba()

        val r = createSimplePipeline(cameraFrameRgba)
        //val r = createAdvancedPipeline(cameraFrameRgba)

        return r
    }

    private fun createSimplePipeline(cameraFrameRgba: Mat?): Mat {

//        val roi = Mat.zeros(imageHeight, imageWidth, CvType.CV_8UC1)
//        Imgproc.fillPoly(roi, listOf(roiPolygon), Scalar(255.0))
//        val cameraFrameCropped = Mat.zeros(720, 1280, CvType.CV_8UC3)
//        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, cameraFrameCropped, roi)

        //Imgproc.bilateralFilter(cameraFrameHls, cameraFrameCropped,9, 75.0, 75.0)

        extractRoadLanesColors(cameraFrameRgba)

        cameraFrameWhiteYellowMasked!!.release()
        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, cameraFrameWhiteYellowMasked, frameMaskTotal)

        Imgproc.cvtColor(cameraFrameWhiteYellowMasked, cameraFrameGrayScale, Imgproc.COLOR_BGR2GRAY, 3)
        Imgproc.GaussianBlur(cameraFrameGrayScale, cameraFrameGrayScale, Size(5.0, 5.0), 0.0, 0.0)
        //Imgproc.equalizeHist(cameraFrameGrayScale, cameraFrameGrayScale)

        val roi = Mat.zeros(imageHeight, imageWidth, CvType.CV_8UC1)
        Imgproc.fillPoly(roi, listOf(roiPolygon), Scalar(255.0))
        cameraFrameCropped = Mat.zeros(imageHeight, imageWidth, CvType.CV_8UC3)
        Core.bitwise_and(cameraFrameGrayScale, cameraFrameGrayScale, cameraFrameCropped, roi)

        applyCanny()
        applyHoughTransform()
        separateLines(allLines)
        chooseExtrapolationRenderingMode(cameraFrameRgba)
        drawRoi(cameraFrameRgba)
        drawTextInfo(cameraFrameRgba)

        roi.release()
        //processedFrameCroppedIntoRoi.release()
        //cameraFrameCropped.release()

        return when (cameraMode) {
            0 -> cameraFrameHls!!
            1 -> cameraFrameHsv!!
            2 -> cameraFrameWhiteYellowMasked!!
            3 -> cameraFrameGrayScale!!
            4 -> cameraFrameCropped!!
            else -> cameraFrameRgba!!
        }
    }

    private fun extractRoadLanesColors(cameraFrameRgba: Mat?) {
        //HLS
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHls, Imgproc.COLOR_RGB2HLS, 3)
        Core.inRange(cameraFrameHls, Scalar(0.0, 180.0, 0.0), Scalar(180.0, 255.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHls, Scalar(15.0, 38.0, 115.0), Scalar(35.0, 204.0, 255.0), frameMaskYellow)
        val hlsMask = Mat()
        Core.bitwise_or(frameMaskWhite, frameMaskYellow, hlsMask)

        //HSV
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHsv, Imgproc.COLOR_RGB2HSV, 3)
        Core.inRange(cameraFrameHsv, Scalar(18.0, 0.0, 180.0), Scalar(255.0, 80.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHsv, Scalar(0.0, 100.0, 100.0), Scalar(50.0, 255.0, 255.0), frameMaskYellow)
        val hsvMask = Mat()
        Core.bitwise_or(frameMaskWhite, frameMaskYellow, hsvMask)

        if (allLines.size < 10) {
            Core.bitwise_or(hlsMask, hsvMask, frameMaskTotal)
        } else {
            Core.bitwise_and(hlsMask, hsvMask, frameMaskTotal)
        }

        hlsMask.release()
        hsvMask.release()
    }

    private fun drawHoughLines(cameraFrameRgba: Mat?) {
        leftLines.forEach {
            Imgproc.line(
                cameraFrameRgba,
                Point(it.xStart, it.yStart),
                Point(it.xEnd, it.yEnd),
                Scalar(0.0, 0.0, 255.0),
                3,
                Imgproc.LINE_AA
            )
        }
        rightLines.forEach {
            Imgproc.line(
                cameraFrameRgba,
                Point(it.xStart, it.yStart),
                Point(it.xEnd, it.yEnd),
                Scalar(0.0, 255.0, 100.0),
                3,
                Imgproc.LINE_AA
            )
        }
    }

    private fun applyCanny() {
        val mean = Core.mean(cameraFrameCropped)
        val sigma = 0.33
        val lowerBound = max(150.0, (1.0 - sigma) * mean.`val`[0])
        val upperBound = max(255.0, (1.0 + sigma) * mean.`val`[0])
        Imgproc.Canny(cameraFrameCropped, cameraFrameCropped, lowerBound, upperBound)

    }

    private fun applyHoughTransform() {
        val linesP = Mat()
        Imgproc.HoughLinesP(cameraFrameCropped, linesP, 1.0, Math.PI / 180, 40, 10.0, 50.0)

        var lineCoords: DoubleArray?
        allLines.clear()
        for (i in 0..linesP.rows()) {
            lineCoords = linesP.get(i, 0)
            if (lineCoords != null) {
                allLines.add(Line(lineCoords[0], lineCoords[1], lineCoords[2], lineCoords[3]))
            }
        }

        val newLines = allLines.filter { it.getSlope() != null }.toCollection(LinkedList())
        allLines = newLines

        linesP.release()
    }

    private fun drawExtrapolatedLine(cameraFrame: Mat?, lines: MutableList<Line>, drawOnTheLeft: Boolean) {
        val roiPolygonPoints = roiPolygon!!.toArray()
        if (drawOnTheLeft) {
            if (lines.size > 0) {
                computeLinearExtrapolationFormula(lines)
                if (leftCoefs.size >= 10) leftCoefs.pollLast()
                leftCoefs.offerFirst(Pair(regression.slope, regression.intercept))
            } else {
                leftCoefs.pollLast()
            }
        } else {
            if (lines.size > 0) {
                computeLinearExtrapolationFormula(lines)
                if (rightCoefs.size >= 10) rightCoefs.pollLast()
                rightCoefs.offerFirst(Pair(regression.slope, regression.intercept))
            } else {
                rightCoefs.pollLast()
            }
        }
        val imageMidpoint = IMAGE_MIDPOINT
        if (drawOnTheLeft) {
            val avgSlope = leftCoefs.map { it.first }.sum() / leftCoefs.size
            val avgIntercept = leftCoefs.map { it.second }.sum() / leftCoefs.size
            Imgproc.line(
                cameraFrame,
                Point(roiPolygonPoints[0].x, avgSlope * roiPolygonPoints[0].x + avgIntercept),
                Point(imageMidpoint - 1.0, avgSlope * imageMidpoint + avgIntercept),
                Scalar(255.0, 0.0, 0.0),
                3,
                Imgproc.LINE_AA
            )
        } else {
            val avgSlope = rightCoefs.map { it.first }.sum() / rightCoefs.size
            val avgIntercept = rightCoefs.map { it.second }.sum() / rightCoefs.size
            Imgproc.line(
                cameraFrame,
                Point(imageMidpoint + 1.0, avgSlope * (imageMidpoint + 1.0) + avgIntercept),
                Point(roiPolygonPoints[3].x, avgSlope * roiPolygonPoints[3].x + avgIntercept),
                Scalar(255.0, 0.0, 0.0),
                3,
                Imgproc.LINE_AA
            )
        }
    }

    private fun drawExtrapolatedCurve(cameraFrame: Mat?, lines: MutableList<Line>, drawOnTheLeft: Boolean) {
        val roiPolygonPoints = roiPolygon!!.toArray()
        computePolynomialExtrapolationFormula(lines)
        if (obs.toList().size == 0) return
        val coeffs = polyFitter.fit(obs.toList())
        val polynomialFunc = PolynomialFunction(coeffs)

        val imageMidpoint = IMAGE_MIDPOINT
        val polyPoints = ArrayList<Point>()


        if (drawOnTheLeft) {
            for (x in roiPolygonPoints[0].x.toInt() until imageMidpoint) {
                polyPoints.add(Point(x.toDouble(), polynomialFunc.value(x.toDouble())))
            }
            val matOfPolyPoints = MatOfPoint()
            matOfPolyPoints.fromList(polyPoints)

            Imgproc.polylines(
                cameraFrame, listOf(matOfPolyPoints), false, Scalar(255.0, 165.0, 0.0), 3
            )
        } else {
            for (x in imageMidpoint until roiPolygonPoints[3].x.toInt()) {
                polyPoints.add(Point(x.toDouble(), polynomialFunc.value(x.toDouble())))
            }
            val matOfPolyPoints = MatOfPoint()
            matOfPolyPoints.fromList(polyPoints)

            Imgproc.polylines(
                cameraFrame, listOf(matOfPolyPoints), false, Scalar(255.0, 165.0, 0.0), 3
            )
        }
    }

    private fun chooseExtrapolationRenderingMode(cameraFrameRgba: Mat?) {
        when (extrapolationMode) {
            0 -> {
                drawExtrapolatedLine(cameraFrameRgba, leftLines, true)
                drawExtrapolatedLine(cameraFrameRgba, rightLines, false)
            }
            1 -> {
                drawExtrapolatedCurve(cameraFrameRgba, leftLines, true)
                drawExtrapolatedCurve(cameraFrameRgba, rightLines, false)
            }

            2 -> {
                drawExtrapolatedLine(cameraFrameRgba, leftLines, true)
                drawExtrapolatedLine(cameraFrameRgba, rightLines, false)
                drawHoughLines(cameraFrameRgba)
            }
            3 -> {
                drawExtrapolatedLine(cameraFrameRgba, leftLines, true)
                drawExtrapolatedLine(cameraFrameRgba, rightLines, false)
                drawExtrapolatedCurve(cameraFrameRgba, leftLines, true)
                drawExtrapolatedCurve(cameraFrameRgba, rightLines, false)
            }
            4 -> {}
            else -> {
                drawExtrapolatedLine(cameraFrameRgba, leftLines, true)
                drawExtrapolatedLine(cameraFrameRgba, rightLines, false)
                drawExtrapolatedCurve(cameraFrameRgba, leftLines, true)
                drawExtrapolatedCurve(cameraFrameRgba, rightLines, false)
                drawHoughLines(cameraFrameRgba)
            }
        }
    }

    private fun drawTextInfo(cameraFrameRgba: Mat?) {
        Imgproc.putText(
            cameraFrameRgba,
            "l: ${leftLines.size}, r: ${rightLines.size}",
            Point(50.0, 50.0),
            Core.FONT_HERSHEY_COMPLEX,
            1.0,
            Scalar(255.0, 0.0, 0.0)
        )
    }

    private fun drawRoi(cameraFrameRgba: Mat?) {
        Imgproc.polylines(
            cameraFrameRgba, listOf(roiPolygon), false, Scalar(0.0, 255.0, 0.0),
            1, Imgproc.LINE_AA
        )
    }

    private fun separateLines(allLines: MutableList<Line>) {
        leftLines.clear()
        rightLines.clear()
        var slope: Double?
        val imageMidpoint = IMAGE_MIDPOINT
        allLines.forEach {
            slope = it.getSlope()
            if (slope != null) {
                if (it.xStart < imageMidpoint && it.xEnd < imageMidpoint) {
                    leftLines.add(it)
                } else if (it.xStart > imageMidpoint && it.xEnd > imageMidpoint) {
                    rightLines.add(it)
                }
            }
        }
    }

    private fun computeLinearExtrapolationFormula(lines: MutableList<Line>) {
        regression.clear()
        var slope: Double?
        lines.forEach {
            slope = it.getSlope()
            if (slope != null) {
                regression.addData(it.xStart, it.yStart)
                regression.addData(it.xEnd, it.yEnd)
            }
        }
    }

    private fun computePolynomialExtrapolationFormula(lines: MutableList<Line>) {
        obs.clear()
        var slope: Double?
        lines.forEach {
            slope = it.getSlope()
            if (slope != null) {
                obs.add(it.xStart, it.yStart)
                obs.add(it.xEnd, it.yEnd)
            }
        }
    }

    private fun writeHistInfo(hist: IntArray) {
        var id = 0
        var v = 0
        for (i in hist.indices) {
            if (hist[i] > v) {
                v = hist[i]
                id = i
            }
        }
        Imgproc.putText(
            cameraFrameWhiteYellowMasked,
            "Hist max at $id is $v",
            Point(50.0, 50.0),
            Core.FONT_HERSHEY_COMPLEX,
            1.0,
            Scalar(255.0, 0.0, 0.0)
        )
    }

    private fun computeHistogram(combinedImage: Mat): IntArray {
        val arr = IntArray(1280)
        for (i in arr.indices) {
            arr[i] = Core.countNonZero(combinedImage.col(i))
        }

        return arr
    }

    private fun createAdvancedPipeline(cameraFrameRgba: Mat?): Mat {
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHls, Imgproc.COLOR_RGB2HLS, 3)

        Core.inRange(cameraFrameHls, Scalar(0.0, 200.0, 0.0), Scalar(180.0, 255.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHls, Scalar(15.0, 38.0, 115.0), Scalar(35.0, 204.0, 255.0), frameMaskYellow)

        Core.bitwise_or(frameMaskWhite, frameMaskYellow, frameMaskTotal)

        cameraFrameWhiteYellowMasked!!.release()
        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, cameraFrameWhiteYellowMasked, frameMaskTotal)

        perspTransformMat = Imgproc.getPerspectiveTransform(srcPersp, dstPersp)
        Imgproc.warpPerspective(
            cameraFrameWhiteYellowMasked,
            cameraFrameWhiteYellowMasked,
            perspTransformMat,
            previewSize,
            Imgproc.INTER_LINEAR
        )

        val hist = computeHistogram(cameraFrameWhiteYellowMasked!!)
        writeHistInfo(hist)

        return cameraFrameWhiteYellowMasked!!
    }

    private fun converScalarHsv2Rgba(hsvColor: Scalar?): Scalar {
        val pointMatRgba = Mat()
        val pointMatHsv = Mat(1, 1, CvType.CV_8UC3, hsvColor!!)
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4)

        return Scalar(pointMatRgba.get(0, 0))
    }


    companion object {
        private val TAG = "RecognitionActivity"
    }
}