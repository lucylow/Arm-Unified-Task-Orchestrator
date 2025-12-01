//
//  ModelRunner.swift
//  AutoRL
//
//  Singleton class for managing PyTorch model inference on iOS
//  Features: Thread-safe model loading, efficient image processing, performance monitoring
//

import Foundation
import UIKit
import Accelerate

#if canImport(PyTorchMobile)
import PyTorchMobile
#endif

actor ModelRunner {
    static let shared = ModelRunner()
    
    struct InferenceResult {
        let output: String
        let latencyMs: Int64
        let confidence: Float?
    }
    
    private var model: Any? = nil
    private var isModelLoaded = false
    
    private let modelName = "model_mobile_quant.pt"
    private let inputSize: Int = 224
    
    private init() {}
    
    /// Load the PyTorch model from bundle
    /// Thread-safe operation (actor isolation)
    func loadModel() async -> Bool {
        if isModelLoaded && model != nil {
            print("📱 Model already loaded")
            return true
        }
        
        do {
            print("📱 Loading model: \(modelName)")
            
            guard let modelPath = Bundle.main.path(forResource: modelName, ofType: nil) ??
                    Bundle.main.path(forResource: modelName.replacingOccurrences(of: ".pt", with: ""), ofType: "pt") else {
                print("❌ Model file not found: \(modelName)")
                return false
            }
            
            #if canImport(PyTorchMobile)
            // Note: PyTorch Mobile iOS API may differ - adjust based on actual implementation
            // let torchModel = try TorchModule(fileAtPath: modelPath)
            // model = torchModel
            print("⚠️ PyTorch Mobile import available but implementation needs PyTorch Mobile iOS SDK")
            isModelLoaded = true
            return true
            #else
            // Fallback: Use Core ML or mock implementation
            print("⚠️ PyTorch Mobile not available, using mock model")
            isModelLoaded = true
            return true
            #endif
        } catch {
            print("❌ Failed to load model: \(error)")
            isModelLoaded = false
            model = nil
            return false
        }
    }
    
    /// Check if model is loaded
    func isModelLoaded() -> Bool {
        return isModelLoaded && model != nil
    }
    
    /// Run inference on a UIImage
    func runInference(image: UIImage? = nil) async -> InferenceResult? {
        guard isModelLoaded, model != nil else {
            print("❌ Model not loaded")
            return nil
        }
        
        do {
            let startTime = Date()
            
            // Use provided image or load test image
            let inputImage = image ?? loadTestImage()
            guard let inputImage = inputImage else {
                print("❌ Could not load input image")
                return nil
            }
            
            // Preprocess image
            let inputTensor = try preprocessImage(inputImage)
            
            // Run inference
            #if canImport(PyTorchMobile)
            // Note: Actual PyTorch Mobile iOS API implementation needed
            // if let torchModel = model as? TorchModule {
            //     let outputTensor = try torchModel.forward(inputTensor)
            //     let result = postprocessOutput(outputTensor)
            //     let latencyMs = Int64(Date().timeIntervalSince(startTime) * 1000)
            //     return InferenceResult(output: result, latencyMs: latencyMs, confidence: extractConfidence(outputTensor))
            // }
            #endif
            
            // Mock inference for testing (fallback)
            let latencyMs = Int64(Date().timeIntervalSince(startTime) * 1000)
            return InferenceResult(
                output: "Detected: Button (score: 0.85) [Mock]",
                latencyMs: latencyMs,
                confidence: 0.85
            )
        } catch {
            print("❌ Inference failed: \(error)")
            return nil
        }
    }
    
    /// Preprocess UIImage to tensor
    private func preprocessImage(_ image: UIImage) throws -> Any {
        // Resize image to model input size
        guard let resizedImage = image.resized(to: CGSize(width: inputSize, height: inputSize)),
              let cgImage = resizedImage.cgImage else {
            throw NSError(domain: "ModelRunner", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }
        
        // Convert to RGB
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw NSError(domain: "ModelRunner", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create context"])
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Normalize pixel values to [0, 1] and apply ImageNet normalization
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        
        var normalizedData = [Float](repeating: 0, count: width * height * 3)
        
        for i in 0..<(width * height) {
            let r = Float(pixelData[i * 4]) / 255.0
            let g = Float(pixelData[i * 4 + 1]) / 255.0
            let b = Float(pixelData[i * 4 + 2]) / 255.0
            
            normalizedData[i] = (r - mean[0]) / std[0]
            normalizedData[i + width * height] = (g - mean[1]) / std[1]
            normalizedData[i + width * height * 2] = (b - mean[2]) / std[2]
        }
        
        #if canImport(PyTorchMobile)
        // Note: Actual PyTorch Mobile tensor creation API needed
        // let tensor = try IValue.from(normalizedData, size: [1, 3, Int32(height), Int32(width)])
        // return tensor
        #endif
        // Return normalized data for mock
        return normalizedData
    }
    
    /// Postprocess model output to readable result
    private func postprocessOutput(_ output: Any) -> String {
        #if canImport(PyTorchMobile)
        // Note: Actual PyTorch Mobile output parsing needed
        // guard let tensor = output as? IValue,
        //       let tensorData = try? tensor.toTensor().getDataAsFloatArray() else {
        //     return "Failed to parse output"
        // }
        // ... process tensorData ...
        #endif
        
        // Mock output
        let classNames = [
            "Button", "Text", "Image", "Input", "List",
            "Card", "Icon", "Header", "Footer", "Other"
        ]
        return "Detected: \(classNames[0]) (score: 0.85) [Mock]"
    }
    
    /// Extract confidence score from output
    private func extractConfidence(_ output: Any) -> Float? {
        #if canImport(PyTorchMobile)
        // Note: Actual PyTorch Mobile confidence extraction needed
        #endif
        return 0.85
    }
    
    /// Load test image from bundle
    private func loadTestImage() -> UIImage? {
        if let image = UIImage(named: "test_screen") {
            return image
        }
        
        // Create a placeholder image
        let size = CGSize(width: inputSize, height: inputSize)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        UIColor.systemGray.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    /// Cleanup resources
    func cleanup() async {
        #if canImport(PyTorchMobile)
        // PyTorch models are automatically deallocated
        #endif
        model = nil
        isModelLoaded = false
        print("🧹 ModelRunner cleaned up")
    }
}

// MARK: - UIImage Extension
extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

