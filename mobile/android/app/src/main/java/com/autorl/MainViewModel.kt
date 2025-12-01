package com.autorl

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import javax.inject.Inject

/**
 * ViewModel for MainActivity following MVVM architecture.
 * 
 * Manages:
 * - Model loading state
 * - Inference execution
 * - UI state updates
 * - Error handling
 */
@HiltViewModel
class MainViewModel @Inject constructor(
    private val modelRunner: ModelRunner
) : ViewModel() {
    
    sealed class UIState {
        object Idle : UIState()
        object Loading : UIState()
        object ModelLoaded : UIState()
        object RunningInference : UIState()
        data class InferenceComplete(
            val result: String,
            val latencyMs: Long
        ) : UIState()
        data class Error(val message: String) : UIState()
    }
    
    private val _uiState = MutableStateFlow<UIState>(UIState.Idle)
    val uiState: StateFlow<UIState> = _uiState.asStateFlow()
    
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    private val _successMessage = MutableStateFlow<String?>(null)
    val successMessage: StateFlow<String?> = _successMessage.asStateFlow()
    
    init {
        Timber.d("MainViewModel initialized")
    }
    
    /**
     * Initialize and load the model.
     */
    fun initializeModel() {
        viewModelScope.launch {
            try {
                _uiState.value = UIState.Loading
                
                val success = modelRunner.loadModel()
                
                if (success) {
                    _uiState.value = UIState.ModelLoaded
                    _successMessage.value = "Model loaded successfully"
                    Timber.d("Model loaded successfully")
                } else {
                    _uiState.value = UIState.Error("Failed to load model")
                    _errorMessage.value = "Could not load the AI model. Please check if the model file exists."
                }
            } catch (e: Exception) {
                val errorMsg = "Error loading model: ${e.message}"
                _uiState.value = UIState.Error(errorMsg)
                _errorMessage.value = errorMsg
                Timber.e(e, "Failed to load model")
            }
        }
    }
    
    /**
     * Run inference on a test image or bitmap.
     */
    fun runInference() {
        viewModelScope.launch {
            try {
                if (!modelRunner.isModelLoaded()) {
                    _errorMessage.value = "Model not loaded. Please wait for model to load."
                    return@launch
                }
                
                _uiState.value = UIState.RunningInference
                
                val result = modelRunner.runInference()
                
                if (result != null) {
                    _uiState.value = UIState.InferenceComplete(
                        result = result.output,
                        latencyMs = result.latencyMs
                    )
                    _successMessage.value = "Inference completed successfully"
                    Timber.d("Inference completed in ${result.latencyMs}ms")
                } else {
                    _uiState.value = UIState.Error("Inference returned no result")
                    _errorMessage.value = "Inference failed. Please try again."
                }
            } catch (e: Exception) {
                val errorMsg = "Error during inference: ${e.message}"
                _uiState.value = UIState.Error(errorMsg)
                _errorMessage.value = errorMsg
                Timber.e(e, "Inference failed")
            }
        }
    }
    
    /**
     * Mark error message as shown.
     */
    fun errorMessageShown() {
        _errorMessage.value = null
    }
    
    /**
     * Mark success message as shown.
     */
    fun successMessageShown() {
        _successMessage.value = null
    }
    
    /**
     * Cleanup resources.
     */
    fun cleanup() {
        viewModelScope.launch {
            modelRunner.cleanup()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        cleanup()
        Timber.d("MainViewModel cleared")
    }
}

