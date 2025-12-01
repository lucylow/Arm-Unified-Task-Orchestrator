package com.autorl

import android.app.Application
import dagger.hilt.android.HiltAndroidApp
import timber.log.Timber

/**
 * Application class for AutoRL Android app.
 * Initializes dependency injection and logging.
 */
@HiltAndroidApp
class AutoRLApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize logging
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        } else {
            // In release, you might want to use a crash reporting service
            // For now, use a minimal tree that only logs errors
            Timber.plant(object : Timber.Tree() {
                override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
                    // Only log errors in release builds
                    if (priority >= android.util.Log.ERROR) {
                        android.util.Log.e(tag, message, t)
                    }
                }
            })
        }
        
        Timber.d("AutoRL Application initialized")
    }
}

