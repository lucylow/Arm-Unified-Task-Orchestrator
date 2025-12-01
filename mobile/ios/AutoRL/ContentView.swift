//
//  ContentView.swift
//  AutoRL
//
//  Main Content View - SwiftUI equivalent of MainActivity
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = MainViewModel()
    
    var body: some View {
        NavigationView {
            MainView(viewModel: viewModel)
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
}

struct MainView: View {
    @ObservedObject var viewModel: MainViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header Card
                HeaderCard()
                
                // Status Card
                StatusCard(
                    status: viewModel.statusText,
                    isLoading: viewModel.isLoading
                )
                
                // Results Card
                ResultsCard(
                    result: viewModel.resultText,
                    latency: viewModel.latencyText
                )
                
                // Action Buttons
                ActionButtons(
                    onStartTask: { viewModel.runInference() },
                    onRetry: { viewModel.initializeModel() },
                    isStartEnabled: viewModel.isStartButtonEnabled,
                    showRetry: viewModel.showRetryButton
                )
            }
            .padding()
        }
        .navigationTitle("AutoRL")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            viewModel.initializeModel()
        }
        .alert("Error", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.dismissError() } }
        )) {
            Button("OK") {
                viewModel.dismissError()
            }
        } message: {
            if let error = viewModel.errorMessage {
                Text(error)
            }
        }
        .alert("Success", isPresented: Binding(
            get: { viewModel.successMessage != nil },
            set: { if !$0 { viewModel.dismissSuccess() } }
        )) {
            Button("OK") {
                viewModel.dismissSuccess()
            }
        } message: {
            if let success = viewModel.successMessage {
                Text(success)
            }
        }
    }
}

// MARK: - Header Card
struct HeaderCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("AutoRL")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.primary)
            
            Text("ARM-Powered Task Automation")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
    }
}

// MARK: - Status Card
struct StatusCard: View {
    let status: String
    let isLoading: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Status")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(status)
                .font(.body)
                .foregroundColor(.primary)
            
            if isLoading {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 4, x: 0, y: 2)
    }
}

// MARK: - Results Card
struct ResultsCard: View {
    let result: String
    let latency: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Results")
                .font(.caption)
                .foregroundColor(.secondary)
            
            if !result.isEmpty {
                Text(result)
                    .font(.body)
                    .foregroundColor(.primary)
            } else {
                Text("No results yet")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .italic()
            }
            
            if !latency.isEmpty {
                Text(latency)
                    .font(.caption)
                    .foregroundColor(.blue)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 4, x: 0, y: 2)
    }
}

// MARK: - Action Buttons
struct ActionButtons: View {
    let onStartTask: () -> Void
    let onRetry: () -> Void
    let isStartEnabled: Bool
    let showRetry: Bool
    
    var body: some View {
        VStack(spacing: 12) {
            Button(action: onStartTask) {
                HStack {
                    Image(systemName: "play.fill")
                    Text("Start Task")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(isStartEnabled ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(!isStartEnabled)
            
            if showRetry {
                Button(action: onRetry) {
                    Text("Retry")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(12)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: Color.black.opacity(0.05), radius: 4, x: 0, y: 2)
    }
}

#Preview {
    ContentView()
}

