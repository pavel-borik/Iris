package com.app.pavelb.iris

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.view.View
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {
    private val MY_PERMISSIONS_REQUEST_CAMERA = 10

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            mainStartButton.visibility = View.INVISIBLE
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), MY_PERMISSIONS_REQUEST_CAMERA )
        } else {
            mainPermissionsButton.visibility = View.GONE
            instructionsTextView.text = getString(R.string.instruction)
        }

        mainPermissionsButton.setOnClickListener {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), MY_PERMISSIONS_REQUEST_CAMERA )
        }

        mainStartButton.setOnClickListener {
            val i = Intent(this, RecognitionActivity::class.java)
            startActivity(i)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        when (requestCode) {
            MY_PERMISSIONS_REQUEST_CAMERA -> {
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    instructionsTextView.text = getString(R.string.instruction)
                    mainStartButton.visibility = View.VISIBLE
                    mainPermissionsButton.visibility = View.GONE
                } else {
                    instructionsTextView.text = getString(R.string.instruction_no_permissions)
                }
                return
            }
            else -> {}
        }
    }
}
