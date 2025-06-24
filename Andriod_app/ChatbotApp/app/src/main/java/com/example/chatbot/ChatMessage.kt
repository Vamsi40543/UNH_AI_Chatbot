package com.teamrivals.bytecatchatbot

//  Model class representing a single message in the chat UI
data class ChatMessage(
    val text: String,     // The content of the message (either from user or  bot)
    val isUser: Boolean   // true if sent by the user, false if sent by the chatbot
)


