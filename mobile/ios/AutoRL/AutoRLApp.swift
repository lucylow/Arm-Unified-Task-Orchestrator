//
//  AutoRLApp.swift
//  AutoRL
//
//  iOS Application Entry Point
//  Initializes the app and sets up dependency injection
//

import SwiftUI

@main
struct AutoRLApp: App {
    @StateObject private var appState = AppState()
    
    init() {
        // Configure logging
        setupLogging()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
    }
    
    private func setupLogging() {
        #if DEBUG
        print("🚀 AutoRL iOS App initialized")
        #endif
    }
}

// Global app state
class AppState: ObservableObject {
    @Published var isModelLoaded = false
    @Published var currentTask: Task?
}

