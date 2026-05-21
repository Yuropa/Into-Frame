//
//  ContentView.swift
//  InfoFrame
//
//  Created by Josh Ford on 5/21/26.
//

import SwiftUI

struct ContentView: View {

    var body: some View {
        VStack {
            Text("Into Frame")
                .font(.largeTitle)
                .bold()

            ToggleImmersiveSpaceButton()
        }
        .padding()
    }
}

#Preview {
    ContentView()
        .environment(AppModel())
}
