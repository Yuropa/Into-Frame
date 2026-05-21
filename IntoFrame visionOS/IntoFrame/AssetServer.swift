import Foundation

struct AssetServer: Sendable {
    let baseURL: URL

    private static func configuredBaseURL() -> URL {
        if let dict = Bundle.main.infoDictionary,
           let urlString = dict["AssetBaseURL"] as? String,
           let url = URL(string: urlString) {
            return url
        }
        return URL(string: "http://localhost:3000/assets/")!
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
