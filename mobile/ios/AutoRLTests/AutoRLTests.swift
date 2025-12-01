//
//  AutoRLTests.swift
//  AutoRLTests
//
//  Unit tests for AutoRL iOS app
//

import XCTest
@testable import AutoRL

final class AutoRLTests: XCTestCase {
    
    var viewModel: MainViewModel!
    
    override func setUp() {
        super.setUp()
        viewModel = MainViewModel()
    }
    
    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }
    
    func testViewModelInitialState() {
        XCTAssertEqual(viewModel.statusText, "Ready to start")
        XCTAssertFalse(viewModel.isLoading)
        XCTAssertFalse(viewModel.isStartButtonEnabled)
        XCTAssertFalse(viewModel.showRetryButton)
    }
    
    func testUtilsFileSizeFormatting() {
        XCTAssertEqual(Utils.getFileSizeString(bytes: 1024), "1.00 KB")
        XCTAssertEqual(Utils.getFileSizeString(bytes: 1048576), "1.00 MB")
        XCTAssertEqual(Utils.getFileSizeString(bytes: 1073741824), "1.00 GB")
    }
    
    func testDeviceInfo() {
        let deviceInfo = Utils.getDeviceInfo()
        XCTAssertNotNil(deviceInfo["device_id"])
        XCTAssertEqual(deviceInfo["os"], "iOS")
        XCTAssertNotNil(deviceInfo["version"])
    }
}

