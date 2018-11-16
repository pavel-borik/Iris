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
import org.opencv.core.*
import org.opencv.utils.Converters

class RecognitionActivity : Activity(), OnTouchListener, CvCameraViewListener2 {

    private var previewSize: Size? = null

    private var cameraFrameRgba: Mat? = null
    private var cameraFrameHls: Mat? = null
    private var cameraFrameResult: Mat? = null
    private var frameMaskWhite: Mat? = null
    private var frameMaskYellow: Mat? = null
    private var frameMaskTotal: Mat? = null
    private var combinedImage: Mat? = null
    private var perspTransformMat: Mat? = null

    private var srcPersp: Mat? = null
    private var dstPersp: Mat? = null

    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView!!.enableView()
                    mOpenCvCameraView!!.setOnTouchListener(this@RecognitionActivity)
                    mOpenCvCameraView!!.setMaxFrameSize(1280, 720)
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
        previewSize = Size(1280.0, 720.0)
        cameraFrameRgba = Mat(height, width, CvType.CV_8UC4)
        cameraFrameHls = Mat(height, width, CvType.CV_8UC4)
        cameraFrameResult = Mat(height, width, CvType.CV_8UC4)
        frameMaskWhite = Mat(height, width, CvType.CV_8UC1)
        frameMaskYellow = Mat(height, width, CvType.CV_8UC1)
        frameMaskTotal = Mat(height, width, CvType.CV_8UC1)
        combinedImage = Mat(height, width, CvType.CV_8UC3)
        perspTransformMat = Mat()
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

        val r = createPipelineSimple(cameraFrameRgba)
        //val r = createPipelineAdvanced(cameraFrameRgba)

        return r
    }

    private fun createPipelineSimple(cameraFrameRgba: Mat?): Mat {
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHls, Imgproc.COLOR_RGB2HLS, 3)

        Core.inRange(cameraFrameHls, Scalar(0.0, 0.0, 0.0), Scalar(180.0, 255.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHls, Scalar(15.0, 38.0, 115.0), Scalar(35.0, 204.0, 255.0), frameMaskYellow)
        Core.bitwise_or(frameMaskWhite, frameMaskYellow, frameMaskTotal)
        combinedImage!!.release()
        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, combinedImage, frameMaskTotal)

        Imgproc.cvtColor(combinedImage, cameraFrameResult, Imgproc.COLOR_RGB2GRAY, 4)
        Imgproc.GaussianBlur(cameraFrameResult, cameraFrameResult, Size(5.0, 5.0), 0.0, 0.0)
        Imgproc.Canny(cameraFrameResult, cameraFrameResult, 50.0, 150.0)

        val roi = Mat.zeros(720, 1280, CvType.CV_8UC1)
        Imgproc.fillPoly(
            roi,
            listOf(MatOfPoint(Point(190.0, 720.0), Point(582.0, 457.0), Point(701.0, 457.0), Point(1145.0, 720.0))),
            Scalar(255.0)
        )
        val result = Mat.zeros(720, 1280, CvType.CV_8UC4)
        Core.bitwise_and(cameraFrameResult, cameraFrameResult, result, roi)

        val linesP = Mat()
        Imgproc.HoughLinesP(result, linesP, 1.0, Math.PI / 180, 15, 20.0, 10.0)

        var points: DoubleArray?
        for (i in 0..linesP.rows()) {
            points = linesP.get(i, 0)
            if (points != null) {
                Imgproc.line(
                    cameraFrameRgba,
                    Point(points[0], points[1]),
                    Point(points[2], points[3]),
                    Scalar(0.0, 0.0, 255.0),
                    3,
                    Imgproc.LINE_AA
                )
            }
        }

        roi.release()
        result.release()
        linesP.release()
        return cameraFrameRgba!!
    }

    private fun createPipelineAdvanced(cameraFrameRgba: Mat?): Mat {
        Imgproc.cvtColor(cameraFrameRgba, cameraFrameHls, Imgproc.COLOR_RGB2HLS, 3)

        Core.inRange(cameraFrameHls, Scalar(0.0, 200.0, 0.0), Scalar(180.0, 255.0, 255.0), frameMaskWhite)
        Core.inRange(cameraFrameHls, Scalar(15.0, 38.0, 115.0), Scalar(35.0, 204.0, 255.0), frameMaskYellow)

        Core.bitwise_or(frameMaskWhite, frameMaskYellow, frameMaskTotal)

        combinedImage!!.release()
        Core.bitwise_and(cameraFrameRgba, cameraFrameRgba, combinedImage, frameMaskTotal)

        perspTransformMat = Imgproc.getPerspectiveTransform(srcPersp, dstPersp)
        Imgproc.warpPerspective(combinedImage, combinedImage, perspTransformMat, previewSize, Imgproc.INTER_LINEAR)

        val hist = computeHistogram(combinedImage!!)
        writeHistInfo(hist)

        return combinedImage!!
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
            combinedImage,
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