package com.autorl

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import dagger.hilt.android.qualifiers.ApplicationContext
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Singleton class for managing PyTorch model inference.
 * 
 * Features:
 * - Thread-safe model loading
 * - Efficient bitmap-to-tensor conversion
 * - Performance monitoring
 * - Resource cleanup
 * - Error handling
 */
@Singleton
class ModelRunner @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    data class InferenceResult(
        val output: String,
        val latencyMs: Long,
        val confidence: Float? = null
    )
    
    private var model: Module? = null
    private val modelLock = Any()
    private var isModelLoaded = false
    
    companion object {
        private const val MODEL_NAME = "model_mobile_quant.pt"
        private const val INPUT_SIZE = 224
        private const val TAG = "ModelRunner"
    }
    
    /**
     * Load the PyTorch model from assets.
     * Thread-safe operation.
     */
    fun loadModel(): Boolean {
        synchronized(modelLock) {
            if (isModelLoaded && model != null) {
                Timber.d("Model already loaded")
                return true
            }
            
            return try {
                Timber.d("Loading model: $MODEL_NAME")
                
                val modelPath = assetFilePath(context, MODEL_NAME)
                if (modelPath == null || !File(modelPath).exists()) {
                    Timber.e("Model file not found: $MODEL_NAME")
                    return false
                }
                
                model = Module.load(modelPath)
                isModelLoaded = true
                
                Timber.d("Model loaded successfully from: $modelPath")
                true
            } catch (e: Exception) {
                Timber.e(e, "Failed to load model")
                isModelLoaded = false
                model = null
                false
            }
        }
    }
    
    /**
     * Check if model is loaded.
     */
    fun isModelLoaded(): Boolean {
        synchronized(modelLock) {
            return isModelLoaded && model != null
        }
    }
    
    /**
     * Run inference on a bitmap image.
     */
    fun runInference(bitmap: Bitmap? = null): InferenceResult? {
        synchronized(modelLock) {
            if (!isModelLoaded || model == null) {
                Timber.e("Model not loaded")
                return null
            }
            
            return try {
                val startTime = System.currentTimeMillis()
                
                // Use provided bitmap or load test image
                val inputBitmap = bitmap ?: loadTestImage()
                if (inputBitmap == null) {
                    Timber.e("Could not load input image")
                    return null
                }
                
                // Preprocess image
                val inputTensor = preprocessBitmap(inputBitmap)
                
                // Run inference
                val outputTensor = model!!.forward(IValue.from(inputTensor)).toTensor()
                
                // Postprocess output
                val result = postprocessOutput(outputTensor)
                
                val latencyMs = System.currentTimeMillis() - startTime
                
                Timber.d("Inference completed in ${latencyMs}ms")
                
                InferenceResult(
                    output = result,
                    latencyMs = latencyMs,
                    confidence = extractConfidence(outputTensor)
                )
            } catch (e: Exception) {
                Timber.e(e, "Inference failed")
                null
            }
        }
    }
    
    /**
     * Preprocess bitmap to tensor.
     */
    private fun preprocessBitmap(bitmap: Bitmap): Tensor {
        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )
        
        // Convert to tensor using PyTorch's utility
        return TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
    }
    
    /**
     * Postprocess model output to readable result.
     */
    private fun postprocessOutput(outputTensor: Tensor): String {
        val scores = outputTensor.dataAsFloatArray
        
        // Find the class with highest score
        var maxIndex = 0
        var maxScore = scores[0]
        
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxIndex = i
            }
        }
        
        // Map index to class name (simplified - adjust based on your model)
        val classNames = arrayOf(
            "Button", "Text", "Image", "Input", "List",
            "Card", "Icon", "Header", "Footer", "Other"
        )
        
        val className = if (maxIndex < classNames.size) {
            classNames[maxIndex]
        } else {
            "Class_$maxIndex"
        }
        
        return "Detected: $className (score: ${String.format("%.2f", maxScore)})"
    }
    
    /**
     * Extract confidence score from output tensor.
     */
    private fun extractConfidence(outputTensor: Tensor): Float? {
        return try {
            val scores = outputTensor.dataAsFloatArray
            scores.maxOrNull()
        } catch (e: Exception) {
            Timber.e(e, "Failed to extract confidence")
            null
        }
    }
    
    /**
     * Load test image from assets.
     */
    private fun loadTestImage(): Bitmap? {
        return try {
            val inputStream = context.assets.open("test_screen.png")
            BitmapFactory.decodeStream(inputStream)
        } catch (e: Exception) {
            Timber.w(e, "Test image not found, using placeholder")
            // Create a placeholder bitmap
            Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        }
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        synchronized(modelLock) {
            model?.close()
            model = null
            isModelLoaded = false
            Timber.d("ModelRunner cleaned up")
        }
    }
}

