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
import com.app.pavelb.iris.utils.Line
import org.apache.commons.math3.fitting.PolynomialCurveFitter
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.opencv.core.*
import org.opencv.utils.Converters
import java.util.*

class RecognitionActivity : Activity(), OnTouchListener, CvCameraViewListener2 {
    private val imageWidth: Int = 1280
    private val imageHeight: Int = 720
    private var previewSize: Size? = null
    private var roiPolygon: MatOfPoint? = null
    private var cameraFrameRgba: Mat? = null
    private var cameraFrameHls: Mat? = null
    private var cameraFrameGrayScale: Mat? = null
    private var frameMaskWhite: Mat? = null
    private var frameMaskYellow: Mat? = null
    private var frameMaskTotal: Mat? = null
    private var cameraFrameWhiteYellowMasked: Mat? = null
    private var perspTransformMat: Mat? = null
    private var srcPersp: Mat? = null
    private var dstPersp: Mat? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private val allLines: MutableList<Line> = LinkedList()
    private val leftLines: MutableList<Line> = LinkedList()
    private val rightLines: MutableList<Line> = LinkedList()


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
        cameraFrameRgba = Mat(height, width, CvType.CV_8UC4)
        cameraFrameHls = Mat(height, width, CvType.CV_8UC4)
        cameraFrameGrayScale = Mat(height, width, CvType.CV_8UC4)
        frameMaskWhite = Mat(height, width, CvType.CV_8UC1)
        frameMaskYellow = Mat(height, width, CvType.CV_8UC1)
        frameMaskTotal = Mat(height, width, CvType.CV_8UC1)
        cameraFrameWhiteYellowMasked = Mat(height, width, CvType.CV_8UC3)
        perspTransformMat = Mat()
        roiPolygon = MatOfPoint(Point(190.0, 720.0), Point(582.0, 457.0), Point(701.0, 457.0), Point(1145.0, 720.0))

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
        val cols = cameraFrameRgba!!.cols()
        val rows = cameraFrameRgba!!.rows()

        val xOffset = (mOpenCvCameraView!!.width - cols) / 2
        val yOffset = (mOpenCvCameraView!!.height - rows) / 2

        val x = event.x.toInt() - xOffset
        val y = event.y.toInt() - yOffset

        Log.i(TAG, "Touch image coordinates: ($x, $y)")

        return false // don't need subsequent touch events
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat? {
        cameraFrameRgba = inputFrame.rgba()

        val r = createSimplePipeline(cameraFrameRgba)
        //val r = createAdvancedPipeline(cameraFrameRgba)

        return r
    }

    private fun createSimplePipeline(cameraFrameRgba: Mat?): Mat {
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHls, Imgproc.COLOR_RGB2HLS, 3)

        Core.inRange(cameraFrameHls, Scalar(0.0, 200.0, 0.0), Scalar(180.0, 255.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHls, Scalar(15.0, 38.0, 115.0), Scalar(35.0, 204.0, 255.0), frameMaskYellow)
        Core.bitwise_or(frameMaskWhite, frameMaskYellow, frameMaskTotal)
        cameraFrameWhiteYellowMasked!!.release()
        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, cameraFrameWhiteYellowMasked, frameMaskTotal)

        Imgproc.cvtColor(cameraFrameWhiteYellowMasked, cameraFrameGrayScale, Imgproc.COLOR_RGB2GRAY, 4)
        Imgproc.GaussianBlur(cameraFrameGrayScale, cameraFrameGrayScale, Size(5.0, 5.0), 0.0, 0.0)
        Imgproc.Canny(cameraFrameGrayScale, cameraFrameGrayScale, 50.0, 150.0)

        val roi = Mat.zeros(imageHeight, imageWidth, CvType.CV_8UC1)
        Imgproc.fillPoly(
            roi,
            listOf(roiPolygon),
            Scalar(255.0)
        )
        showRoi(cameraFrameRgba)

        val processedFrameCroppedIntoRoi = Mat.zeros(720, 1280, CvType.CV_8UC4)
        Core.bitwise_and(cameraFrameGrayScale, cameraFrameGrayScale, processedFrameCroppedIntoRoi, roi)

        val linesP = Mat()
        Imgproc.HoughLinesP(processedFrameCroppedIntoRoi, linesP, 1.0, Math.PI / 180, 15, 10.0, 10.0)

        var lineCoords: DoubleArray?
        allLines.clear()
        for (i in 0..linesP.rows()) {
            lineCoords = linesP.get(i, 0)
            if (lineCoords != null) {
                allLines.add(Line(lineCoords[0], lineCoords[1], lineCoords[2], lineCoords[3]))
            }
        }

        separateLines(allLines)

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
                Scalar(255.0, 0.0, 0.0),
                3,
                Imgproc.LINE_AA
            )
        }

        Imgproc.putText(
            cameraFrameRgba,
            "l: ${leftLines.size}, r: ${rightLines.size}",
            Point(50.0, 50.0),
            Core.FONT_HERSHEY_COMPLEX,
            1.0,
            Scalar(255.0, 0.0, 0.0)
        )

        roi.release()
        processedFrameCroppedIntoRoi.release()
        linesP.release()
        return cameraFrameRgba!!
    }

    private fun showRoi(cameraFrameRgba: Mat?) {
        Imgproc.polylines(
            cameraFrameRgba, listOf(roiPolygon), false, Scalar(0.0, 255.0, 0.0),
            1, Imgproc.LINE_AA
        )
    }

    private fun separateLines(allLines: MutableList<Line>) {
        leftLines.clear()
        rightLines.clear()
        var slope: Double?
        val imageMidpoint = imageWidth / 2
        allLines.forEach {
            slope = it.getSlope()
            if (slope != null) {
                if (slope!! > 0.0 && it.xStart < imageMidpoint && it.xEnd < imageMidpoint) {
                    leftLines.add(it)
                } else if (slope!! < 0.0 && it.xStart > imageMidpoint && it.xEnd > imageMidpoint) {
                    rightLines.add(it)
                }
            }
        }
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