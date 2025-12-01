// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AutoRL",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "AutoRL",
            targets: ["AutoRL"]),
    ],
    dependencies: [
        // PyTorch Mobile for iOS
        // Note: PyTorch Mobile needs to be added via CocoaPods or manually
        // For now, we'll use a conditional compilation approach
    ],
    targets: [
        .target(
            name: "AutoRL",
            dependencies: [],
            path: "AutoRL"
        ),
        .testTarget(
            name: "AutoRLTests",
            dependencies: ["AutoRL"],
            path: "AutoRLTests"
        ),
    ]
)

