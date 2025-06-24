// Package declaration
package com.teamrivals.bytecatchatbot

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.ClickableText
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusEvent
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime

// === Utility to clean HTML tags from chatbot responses ===
fun removeHtmlTags(input: String): String {
    return input.replace(Regex("<.*?>"), "")
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ChatScreen() {
    // Create API instance and coroutine scope
    val chatApi = remember { ChatApi.create() }
    val scope = rememberCoroutineScope()

    // State for user input and list of messages
    var input by remember { mutableStateOf("") }
    val messages = remember { mutableStateListOf<ChatMessage>() }
    val uriHandler = LocalUriHandler.current
    val listState = rememberLazyListState()

    // For editing a user's previous message
    var editingIndex by remember { mutableStateOf<Int?>(null) }
    var editingText by remember { mutableStateOf("") }

    // === First message shown when chatbot loads ===
    LaunchedEffect(Unit) {
        messages.add(ChatMessage("👋 Welcome to ByteCat - advising chatbot for UNHM computing graduate programs. What can I do for you today?", false))
    }

    // === Auto-scroll to the newest message ===
    LaunchedEffect(messages.size) {
        scope.launch {
            delay(250)
            if (messages.isNotEmpty()) {
                listState.animateScrollToItem(messages.lastIndex)
            }
        }
    }

    // === Main Chat UI ===
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        // === Top AppBar Header ===
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF002B7D))
                .padding(vertical = 12.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("ByteCat – Advising Chatbot", color = Color.White, fontSize = 18.sp)
        }

        // === Chat message list ===
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.ime)
                .padding(horizontal = 8.dp),
            verticalArrangement = Arrangement.Top
        ) {
            itemsIndexed(messages) { index, msg ->
                val isUser = msg.isUser
                val bubbleColor = Color(0xFFF5F5F5)
                val bringIntoViewRequester = remember { BringIntoViewRequester() }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    contentAlignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
                ) {
                    // === Editable bubble if user is editing a message ===
                    if (editingIndex == index && isUser) {
                        Column(
                            horizontalAlignment = Alignment.End,
                            modifier = Modifier
                                .background(bubbleColor, shape = RoundedCornerShape(16.dp))
                                .padding(12.dp)
                                .bringIntoViewRequester(bringIntoViewRequester)
                        ) {
                            // TextField for editing
                            TextField(
                                value = editingText,
                                onValueChange = { editingText = it },
                                modifier = Modifier
                                    .widthIn(min = 100.dp, max = 280.dp)
                                    .onFocusEvent {
                                        if (it.isFocused) {
                                            scope.launch {
                                                delay(250)
                                                bringIntoViewRequester.bringIntoView()
                                            }
                                        }
                                    },
                                textStyle = TextStyle(fontSize = 16.sp),
                                colors = TextFieldDefaults.textFieldColors(
                                    backgroundColor = Color.Transparent,
                                    unfocusedIndicatorColor = Color.Transparent,
                                    focusedIndicatorColor = Color.Transparent
                                )
                            )

                            // Action buttons (Cancel and Save)
                            Row(
                                horizontalArrangement = Arrangement.End,
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                TextButton(onClick = {
                                    editingIndex = null
                                    editingText = ""
                                }) {
                                    Text("Cancel", color = Color.Gray)
                                }

                                Button(
                                    onClick = {
                                        messages[index] = ChatMessage(editingText, true)
                                        while (messages.size > index + 1) messages.removeLast()
                                        editingIndex = null
                                        scope.launch {
                                            try {
                                                val response = chatApi.sendMessage(MessageRequest(editingText))
                                                val cleanText = removeHtmlTags(response.response)
                                                    .replace(Regex("^[-•]\\s*"), "")
                                                messages.add(ChatMessage(cleanText, false))
                                            } catch (e: Exception) {
                                                messages.add(ChatMessage("Error: ${e.message ?: "unknown error"}", false))
                                            }
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(backgroundColor = Color(0xFF002B7D))
                                ) {
                                    Text("Save", color = Color.White)
                                }
                            }
                        }
                    } else {
                        // === Regular (non-editing) chat bubble ===
                        Column(horizontalAlignment = Alignment.End) {
                            // === Format links and emails in message ===
                            val annotatedText = buildAnnotatedString {
                                val cleanText = msg.text.replace(Regex("^[-•]\\s*"), "")
                                append(cleanText)

                                val regex = Regex("(https?://[\\w./?=&%-]+)|(\\b[\\w.%+-]+@[\\w.-]+\\.[a-zA-Z]{2,}\\b)")
                                regex.findAll(cleanText).forEach { result ->
                                    val annotation = result.value
                                    val start = result.range.first
                                    val end = result.range.last + 1

                                    addStyle(
                                        style = SpanStyle(
                                            color = Color.Blue,
                                            textDecoration = TextDecoration.Underline
                                        ),
                                        start = start,
                                        end = end
                                    )

                                    val tag = if (annotation.contains("@")) "EMAIL" else "URL"
                                    addStringAnnotation(
                                        tag = tag,
                                        annotation = annotation,
                                        start = start,
                                        end = end
                                    )
                                }
                            }

                            // === Display clickable message bubble ===
                            ClickableText(
                                text = annotatedText,
                                style = TextStyle(fontSize = 16.sp),
                                onClick = { offset ->
                                    annotatedText.getStringAnnotations(start = offset, end = offset)
                                        .firstOrNull()?.let { annotation ->
                                            when (annotation.tag) {
                                                "URL" -> uriHandler.openUri(annotation.item)
                                                "EMAIL" -> uriHandler.openUri("mailto:${annotation.item}")
                                            }
                                        }
                                },
                                modifier = Modifier
                                    .widthIn(
                                        max = if (isUser)
                                            (LocalConfiguration.current.screenWidthDp * 0.85).dp
                                        else
                                            (LocalConfiguration.current.screenWidthDp * 0.7).dp
                                    )
                                    .background(bubbleColor, shape = RoundedCornerShape(16.dp))
                                    .padding(16.dp)
                            )

                            // === Edit button for user messages ===
                            if (isUser) {
                                Text(
                                    text = "Edit",
                                    color = Color(0xFF002B7D),
                                    modifier = Modifier
                                        .padding(top = 4.dp)
                                        .clickable {
                                            editingIndex = index
                                            editingText = msg.text
                                        }
                                )
                            }
                        }
                    }
                }
            }
        }

        // Input bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFFF2F2F2))
                .padding(10.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.White, shape = RoundedCornerShape(30.dp))
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // User input field
                TextField(
                    value = input,
                    onValueChange = { input = it },
                    placeholder = { Text("Type your text here...") },
                    modifier = Modifier
                        .weight(1f)
                        .padding(end = 8.dp),
                    colors = TextFieldDefaults.textFieldColors(
                        backgroundColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent
                    ),
                    maxLines = 3
                )

                // Send button
                IconButton(
                    onClick = {
                        if (input.isNotBlank()) {
                            val userMessage = input
                            messages.add(ChatMessage(userMessage, true))
                            input = ""
                            scope.launch {
                                try {
                                    val response = chatApi.sendMessage(MessageRequest(userMessage))
                                    val cleanText = removeHtmlTags(response.response)
                                        .replace(Regex("^[-•]\\s*"), "")
                                    messages.add(ChatMessage(cleanText, false))
                                } catch (e: Exception) {
                                    messages.add(ChatMessage("Error: ${e.message ?: "unknown error"}", false))
                                }
                            }
                        }
                    },
                    modifier = Modifier
                        .size(42.dp)
                        .background(Color(0xFF002B7D), shape = RoundedCornerShape(21.dp))
                ) {
                    Icon(Icons.Default.Send, contentDescription = "Send", tint = Color.White)
                }
            }
        }
    }
}


