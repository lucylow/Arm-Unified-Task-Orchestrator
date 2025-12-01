package com.autorl.di

import android.content.Context
import com.autorl.ModelRunner
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module for providing application-wide dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    
    @Provides
    @Singleton
    fun provideModelRunner(
        @ApplicationContext context: Context
    ): ModelRunner {
        return ModelRunner(context)
    }
}

