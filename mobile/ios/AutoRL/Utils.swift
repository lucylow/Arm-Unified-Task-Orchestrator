//
//  Utils.swift
//  AutoRL
//
//  Utility functions for the AutoRL iOS app
//

import Foundation
import UIKit

struct Utils {
    /// Get file size in human-readable format
    static func getFileSizeString(bytes: Int64) -> String {
        let kb = Double(bytes) / 1024.0
        let mb = kb / 1024.0
        let gb = mb / 1024.0
        
        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else if mb >= 1.0 {
            return String(format: "%.2f MB", mb)
        } else if kb >= 1.0 {
            return String(format: "%.2f KB", kb)
        } else {
            return "\(bytes) B"
        }
    }
    
    /// Check if a file exists in bundle
    static func fileExistsInBundle(fileName: String) -> Bool {
        return Bundle.main.path(forResource: fileName, ofType: nil) != nil
    }
    
    /// Get file path from bundle
    static func filePathInBundle(fileName: String) -> String? {
        return Bundle.main.path(forResource: fileName, ofType: nil)
    }
    
    /// Copy file from bundle to documents directory
    static func copyFileFromBundleToDocuments(fileName: String) -> String? {
        guard let bundlePath = Bundle.main.path(forResource: fileName, ofType: nil) else {
            return nil
        }
        
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationPath = documentsPath.appendingPathComponent(fileName)
        
        // Check if file already exists
        if FileManager.default.fileExists(atPath: destinationPath.path) {
            return destinationPath.path
        }
        
        do {
            try FileManager.default.copyItem(atPath: bundlePath, toPath: destinationPath.path)
            return destinationPath.path
        } catch {
            print("❌ Failed to copy file: \(error)")
            return nil
        }
    }
    
    /// Get device information
    static func getDeviceInfo() -> [String: String] {
        var info: [String: String] = [:]
        
        info["device_id"] = UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
        info["model"] = UIDevice.current.model
        info["os"] = "iOS"
        info["version"] = UIDevice.current.systemVersion
        info["name"] = UIDevice.current.name
        
        #if arch(arm64)
        info["cpu_abi"] = "arm64"
        #elseif arch(arm)
        info["cpu_abi"] = "armv7"
        #else
        info["cpu_abi"] = "x86_64"
        #endif
        
        return info
    }
}

