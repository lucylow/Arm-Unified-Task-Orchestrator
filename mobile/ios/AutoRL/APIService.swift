//
//  APIService.swift
//  AutoRL
//
//  Service for communicating with the backend API
//  Handles REST API calls, WebSocket connections, and error handling
//

import Foundation
import Combine

class APIService {
    static let shared = APIService()
    
    private let baseURL: String
    private let session: URLSession
    
    init(baseURL: String? = nil) {
        // Default to localhost for simulator, adjust for device
        #if targetEnvironment(simulator)
        self.baseURL = baseURL ?? "http://localhost:8000"
        #else
        // For physical device, use your computer's IP address
        self.baseURL = baseURL ?? "http://192.168.1.100:8000"
        #endif
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: configuration)
    }
    
    // MARK: - Health Check
    
    func healthCheck() async throws -> HealthResponse {
        let url = URL(string: "\(baseURL)/health")!
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }
    
    // MARK: - Task Execution
    
    func executeTask(instruction: String, deviceId: String? = nil, maxSteps: Int = 10) async throws -> TaskResponse {
        let url = URL(string: "\(baseURL)/api/v1/execute")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ExecuteTaskRequest(
            instruction: instruction,
            device_id: deviceId,
            max_steps: maxSteps,
            use_cloud_planner: false
        )
        
        request.httpBody = try JSONEncoder().encode(body)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIError.invalidResponse
        }
        
        return try JSONDecoder().decode(TaskResponse.self, from: data)
    }
    
    // MARK: - Device Management
    
    func getDevices() async throws -> DevicesResponse {
        let url = URL(string: "\(baseURL)/api/v1/devices")!
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(DevicesResponse.self, from: data)
    }
    
    // MARK: - Analytics
    
    func getAnalytics(startTime: String? = nil, endTime: String? = nil) async throws -> AnalyticsResponse {
        var urlString = "\(baseURL)/api/v1/analytics"
        var components = URLComponents(string: urlString)
        
        if let startTime = startTime, let endTime = endTime {
            components?.queryItems = [
                URLQueryItem(name: "start_time", value: startTime),
                URLQueryItem(name: "end_time", value: endTime)
            ]
        }
        
        guard let url = components?.url else {
            throw APIError.invalidURL
        }
        
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(AnalyticsResponse.self, from: data)
    }
}

// MARK: - Request/Response Models

struct ExecuteTaskRequest: Codable {
    let instruction: String
    let device_id: String?
    let max_steps: Int
    let use_cloud_planner: Bool
}

struct TaskResponse: Codable {
    let task_id: String
    let status: String
    let steps_executed: Int?
    let success: Bool?
    let latency_ms: Int?
    let episode_id: String?
}

struct HealthResponse: Codable {
    let status: String
    let version: String?
    let mode: String?
}

struct Device: Codable {
    let device_id: String
    let model: String?
    let os: String?
    let version: String?
    let cpu_abi: String?
    let status: String?
    let uptime_ms: Int?
}

struct DevicesResponse: Codable {
    let devices: [Device]
}

struct AnalyticsResponse: Codable {
    let tasks_total: Int?
    let tasks_success: Int?
    let tasks_failed: Int?
    let success_rate: Double?
    let avg_execution_time: Double?
}

// MARK: - Errors

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case decodingError
    case networkError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .decodingError:
            return "Failed to decode response"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

