//
//  MainViewModel.swift
//  AutoRL
//
//  ViewModel for MainView following MVVM architecture
//  Manages model loading, inference execution, and UI state
//

import Foundation
import Combine
import SwiftUI

class MainViewModel: ObservableObject {
    // Published properties for SwiftUI binding
    @Published var statusText: String = "Ready to start"
    @Published var isLoading: Bool = false
    @Published var resultText: String = ""
    @Published var latencyText: String = ""
    @Published var isStartButtonEnabled: Bool = false
    @Published var showRetryButton: Bool = false
    @Published var errorMessage: String? = nil
    @Published var successMessage: String? = nil
    
    // Dependencies
    private let modelRunner: ModelRunner
    private let apiService: APIService
    
    // State management
    private var cancellables = Set<AnyCancellable>()
    
    enum UIState {
        case idle
        case loading
        case modelLoaded
        case runningInference
        case inferenceComplete(result: String, latencyMs: Int64)
        case error(message: String)
    }
    
    private var currentState: UIState = .idle {
        didSet {
            updateUIForState(currentState)
        }
    }
    
    init(modelRunner: ModelRunner = ModelRunner.shared,
         apiService: APIService = APIService.shared) {
        self.modelRunner = modelRunner
        self.apiService = apiService
    }
    
    /// Initialize and load the model
    func initializeModel() {
        Task { @MainActor in
            do {
                currentState = .loading
                
                let success = await modelRunner.loadModel()
                
                if success {
                    currentState = .modelLoaded
                    successMessage = "Model loaded successfully"
                    print("✅ Model loaded successfully")
                } else {
                    currentState = .error(message: "Failed to load model")
                    errorMessage = "Could not load the AI model. Please check if the model file exists."
                }
            } catch {
                let errorMsg = "Error loading model: \(error.localizedDescription)"
                currentState = .error(message: errorMsg)
                errorMessage = errorMsg
                print("❌ Failed to load model: \(error)")
            }
        }
    }
    
    /// Run inference on a test image or screenshot
    func runInference() {
        Task { @MainActor in
            do {
                guard modelRunner.isModelLoaded else {
                    errorMessage = "Model not loaded. Please wait for model to load."
                    return
                }
                
                currentState = .runningInference
                
                let result = await modelRunner.runInference()
                
                if let result = result {
                    currentState = .inferenceComplete(
                        result: result.output,
                        latencyMs: result.latencyMs
                    )
                    successMessage = "Inference completed successfully"
                    print("✅ Inference completed in \(result.latencyMs)ms")
                } else {
                    currentState = .error(message: "Inference returned no result")
                    errorMessage = "Inference failed. Please try again."
                }
            } catch {
                let errorMsg = "Error during inference: \(error.localizedDescription)"
                currentState = .error(message: errorMsg)
                errorMessage = errorMsg
                print("❌ Inference failed: \(error)")
            }
        }
    }
    
    /// Dismiss error message
    func dismissError() {
        errorMessage = nil
    }
    
    /// Dismiss success message
    func dismissSuccess() {
        successMessage = nil
    }
    
    /// Update UI based on current state
    private func updateUIForState(_ state: UIState) {
        switch state {
        case .idle:
            statusText = "Ready to start"
            isLoading = false
            isStartButtonEnabled = false
            showRetryButton = false
            resultText = ""
            latencyText = ""
            
        case .loading:
            statusText = "Loading model..."
            isLoading = true
            isStartButtonEnabled = false
            showRetryButton = false
            
        case .modelLoaded:
            statusText = "Model loaded"
            isLoading = false
            isStartButtonEnabled = true
            showRetryButton = false
            
        case .runningInference:
            statusText = "Running inference..."
            isLoading = true
            isStartButtonEnabled = false
            showRetryButton = false
            
        case .inferenceComplete(let result, let latencyMs):
            statusText = "Inference complete"
            isLoading = false
            isStartButtonEnabled = true
            showRetryButton = false
            resultText = result
            latencyText = String(format: "Latency: %lld ms", latencyMs)
            
        case .error(let message):
            statusText = message
            isLoading = false
            isStartButtonEnabled = false
            showRetryButton = true
            resultText = ""
            latencyText = ""
        }
    }
    
    deinit {
        Task {
            await modelRunner.cleanup()
        }
    }
}

