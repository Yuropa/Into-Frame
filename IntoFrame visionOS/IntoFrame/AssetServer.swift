import Foundation

struct AssetServer: Sendable {
    var baseURL: URL

    static let defaultBaseURL = "http://localhost:3000/assets/"

    private static func configuredBaseURL() -> URL {
        if let s = UserDefaults.standard.string(forKey: "assetBaseURL"),
           let url = URL(string: s) { return url }
        if let dict = Bundle.main.infoDictionary,
           let s = dict["AssetBaseURL"] as? String,
           let url = URL(string: s) { return url }
        return URL(string: defaultBaseURL)!
    }

    init(baseURL: URL = AssetServer.configuredBaseURL()) {
        self.baseURL = baseURL
    }

    func fetchResource(_ name: String) async throws -> Data {
        let url = baseURL.appendingPathComponent(name)
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}
