package com.teamrivals.bytecatchatbot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

// Entry point for the Android app  

class MainActivity : ComponentActivity() {

    // onCreate is called when the activity is first created
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set the UI content using Jetpack Compose
        // This loads the ChatScreen Composable as the main interface
        setContent {
            ChatScreen()
        }
    }
}


