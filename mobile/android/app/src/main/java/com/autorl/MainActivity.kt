package com.autorl

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.autorl.databinding.ActivityMainBinding
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * Main Activity for AutoRL Android app.
 * 
 * Features:
 * - MVVM architecture with ViewModel
 * - ViewBinding for type-safe view access
 * - Coroutines for async operations
 * - Lifecycle-aware model management
 * - Material Design 3 UI
 */
@AndroidEntryPoint
class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupUI()
        observeViewModel()
        
        // Initialize model loading
        viewModel.initializeModel()
    }
    
    private fun setupUI() {
        binding.apply {
            // Set up button click listener
            startTaskButton.setOnClickListener {
                viewModel.runInference()
            }
            
            // Set up retry button
            retryButton.setOnClickListener {
                viewModel.initializeModel()
            }
        }
    }
    
    private fun observeViewModel() {
        // Observe UI state
        lifecycleScope.launch {
            viewModel.uiState.collect { state ->
                updateUI(state)
            }
        }
        
        // Observe error messages
        lifecycleScope.launch {
            viewModel.errorMessage.collect { message ->
                message?.let {
                    showError(it)
                    viewModel.errorMessageShown()
                }
            }
        }
        
        // Observe success messages
        lifecycleScope.launch {
            viewModel.successMessage.collect { message ->
                message?.let {
                    showSuccess(it)
                    viewModel.successMessageShown()
                }
            }
        }
    }
    
    private fun updateUI(state: MainViewModel.UIState) {
        binding.apply {
            when (state) {
                is MainViewModel.UIState.Idle -> {
                    progressBar.visibility = View.GONE
                    startTaskButton.isEnabled = true
                    retryButton.visibility = View.GONE
                    statusText.text = getString(R.string.ready_to_start)
                }
                
                is MainViewModel.UIState.Loading -> {
                    progressBar.visibility = View.VISIBLE
                    startTaskButton.isEnabled = false
                    retryButton.visibility = View.GONE
                    statusText.text = getString(R.string.loading_model)
                }
                
                is MainViewModel.UIState.ModelLoaded -> {
                    progressBar.visibility = View.GONE
                    startTaskButton.isEnabled = true
                    retryButton.visibility = View.GONE
                    statusText.text = getString(R.string.model_loaded)
                }
                
                is MainViewModel.UIState.RunningInference -> {
                    progressBar.visibility = View.VISIBLE
                    startTaskButton.isEnabled = false
                    retryButton.visibility = View.GONE
                    statusText.text = getString(R.string.running_inference)
                }
                
                is MainViewModel.UIState.InferenceComplete -> {
                    progressBar.visibility = View.GONE
                    startTaskButton.isEnabled = true
                    retryButton.visibility = View.GONE
                    statusText.text = getString(
                        R.string.inference_complete,
                        state.latencyMs
                    )
                    
                    // Update results
                    resultText.text = state.result
                    latencyText.text = getString(
                        R.string.latency_format,
                        state.latencyMs
                    )
                }
                
                is MainViewModel.UIState.Error -> {
                    progressBar.visibility = View.GONE
                    startTaskButton.isEnabled = false
                    retryButton.visibility = View.VISIBLE
                    statusText.text = state.message
                    resultText.text = ""
                    latencyText.text = ""
                }
            }
        }
    }
    
    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        Timber.e("Error: $message")
    }
    
    private fun showSuccess(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        Timber.d("Success: $message")
    }
    
    override fun onDestroy() {
        super.onDestroy()
        viewModel.cleanup()
    }
}

