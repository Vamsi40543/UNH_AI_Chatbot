package com.teamrivals.bytecatchatbot

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

//  Data model for sending a message to the chatbot API 
data class MessageRequest(
    val message: String  // The user's question or input
)

//  Data model for receiving a chatbot response from the API 
data class MessageResponse(
    val response: String  // The chatbot's  answer
)

//  Retrofit interface for calling the chatbot backend ===
interface ChatApi {

    // POST request to /ask endpoint — sends user's message and receives chatbot's reply
    @POST("ask")
    suspend fun sendMessage(@Body request: MessageRequest): MessageResponse

    companion object {
        // Factory method to create and configure the ChatApi instance
        fun create(): ChatApi {
            return Retrofit.Builder()
                .baseUrl("https://whitemount.sr.unh.edu/Rivals/")  // 🔁 Replace with actual backend server IP/URL
                .addConverterFactory(GsonConverterFactory.create()) // Converts JSON to Kotlin objects and vice versa
                .build()
                .create(ChatApi::class.java)  // Creates the ChatApi implementation
        }
    }
}
