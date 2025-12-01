package com.autorl

import android.content.Context
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Utility functions for the AutoRL Android app.
 */
object Utils {
    
    /**
     * Copy an asset file to app's internal storage and return the file path.
     * This is required for PyTorch Mobile to load models.
     * 
     * @param context Application context
     * @param assetName Name of the asset file
     * @return Absolute path to the copied file, or null if failed
     */
    fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        
        try {
            // Check if file already exists
            if (file.exists() && file.length() > 0) {
                Timber.d("Asset file already exists: ${file.absolutePath}")
                return file.absolutePath
            }
            
            // Copy from assets
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            
            Timber.d("Asset file copied to: ${file.absolutePath}")
            return file.absolutePath
        } catch (e: IOException) {
            Timber.e(e, "Failed to copy asset file: $assetName")
            return null
        }
    }
    
    /**
     * Check if a file exists in assets.
     */
    fun assetExists(context: Context, assetName: String): Boolean {
        return try {
            context.assets.open(assetName).use { true }
        } catch (e: IOException) {
            false
        }
    }
    
    /**
     * Get file size in human-readable format.
     */
    fun getFileSizeString(bytes: Long): String {
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        
        return when {
            gb >= 1.0 -> String.format("%.2f GB", gb)
            mb >= 1.0 -> String.format("%.2f MB", mb)
            kb >= 1.0 -> String.format("%.2f KB", kb)
            else -> "$bytes B"
        }
    }
}

