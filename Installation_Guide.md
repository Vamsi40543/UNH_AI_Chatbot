# Bytecat - UNHM Advising Chatbot

## Installation Guide

##  Prerequisites

- Python 3.8+
- Git
- OpenAI API Key ([get it here](https://platform.openai.com))
- AWS account (for deployment)

---

##  Local Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Spring2025-Team-Rivals.git 
cd Spring2025-Team-Rivals
cd Rivals_Chatbot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
```

- **Mac/Linux**
```bash
source venv/bin/activate
```

- **Windows**
```bash
venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API Key

- **Linux/MacOS**
```bash
export OPENAI_API_KEY="your-api-key"
```

- **Windows**
```cmd
set OPENAI_API_KEY="your-api-key"
```

### 5. Add Your Course Catalog PDFs

Place your files inside the `data/` directory.

In `chatbot.py`, set the path:

pdf_directory = "your data directory"


### 6. Run the Chatbot Locally
```bash
python chatbot.py
```

### 7. Access the Web App

Go to:
```
http://127.0.0.1:80/
```

Ask something like:

- "Tell me about COMP 721"
- "When is orientation?"

---

## Automated Testing

- Use scripts in `automated_testing/` to evaluate the chatbot’s performance.

---

## ☁️ AWS EC2 Deployment Guide

### Requirements

- AWS account with EC2 and S3 access  
- Key pair (.pem) for SSH  
- Open ports: 22 (SSH), 80 (HTTP)

---

### EC2 Setup

#### 1. Create AWS Account

- Go to: https://signin.aws.amazon.com/signup  
- Add billing info and verify email/phone

#### 2. Launch EC2 Instance

- **AMI**: Amazon Linux 2  
- **Instance Type**: t3.2xlarge or higher  
- **Storage**: 100GB  
- **Open ports**: 22 (SSH) and 80 (HTTP)

#### 3. Connect to EC2
```bash
ssh -i "your-key.pem" ec2-user@<your-ec2-public-ip>
```

---

## 🚀 Deployment Steps

### 1. Become Root and Update
```bash
sudo su
yum update -y
```

### 2. Install Git & Python
```bash
yum install git -y
yum install python3-pip -y
```

### 3. Clone Repo
```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Spring2025-Team-Rivals.git Unh
cd Spring2025-Team-Rivals
cd Rivals_Chatbot
```

### 4. Install Requirements
```bash
pip3 install -r requirements.txt
```

### 5. Set Data Directory Path

Find absolute path to `data/`:
```bash
cd data
pwd  # Copy this path
```

Update `chatbot.py`:
```python
pdf_directory = "/home/ec2-user/Rivals_Chatbot/data"
```

Use `nano` or `vi` to edit:
```bash
nano chatbot.py
# OR
vi chatbot.py
```

### 6. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key"
```

### 7. Run the Bot
```bash
python chatbot.py
```

### 8. Open in Browser

Go to:
```
http://<your-ec2-public-ip>:5000/
```

## Deployment & Login Guide to UNH Virtual Machine (Apache Proxy)

## Deployment Steps

###  STEP 1: Login to the UNH VM

Prerequisites:

- UNH VPN connected → https://vpn.unh.edu

- SSH access to VM (e.g., whitemount.sr.unh.edu)

Command:
```bash
ssh your_username@whitemount.sr.unh.edu
```
### STEP 2: Navigate to Your Project Folder
```bash
cd ~/Spring2025-Team-Rivals/Rivals_Chatbot
```
### STEP 3: Git Clone (If Needed)

```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Spring2025-Team-Rivals.git Unh
cd Spring2025-Team-Rivals
cd Rivals_Chatbot
```

### STEP 4: Activate Python Environment

If already set up:
```bash
source venv/bin/activate
```
If not yet created:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install flask-cors
```
### STEP 5: Start the Flask App (on localhost, port 8004)
```bash
python chatbot.py
```
You should see:

 * Running on http://127.0.0.1:8004

### STEP 6: Visit the Public Site

Once Flask is running and Apache reverse proxy is active:

- Go to:https://whitemount.sr.unh.edu/Rivals



##  Full Steps to Build and Install Kotlin-Based Android APK Using VS Code

### STEP 1. Clone the Project
```bash
git clone https://github.com/your-username/ByteCatChatbotApp_Full.git

cd ByteCatChatbotApp_Full
```

### STEP 2. Open in VS Code

Make sure the following VS Code extensions are installed:

- Kotlin Language extension

- Gradle Tasks extension (optional)

### STEP 3. Ensure You Have These Installed:

- JDK 17+

- Android SDK & tools

- Gradle (optional if using gradlew)

Add Android SDK tools to PATH:
```bash
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/tools
export PATH=$PATH:$ANDROID_HOME/tools/bin
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

### STEP 4. Clearing the old APK
Before building APK try to clear the old APK
```bash
./gradlew clear
```
### STEP 5. Build the APK Using Terminal

```bash
./gradlew assembleDebug
```

If ./gradlew is not executable:

```bash
chmod +x gradlew
```

### STEP 6. Locate the APK

After build completes, locate the generated APK here:

- ChatbotApp/app/build/outputs/apk/debug/app-debug.apk

### STEP 7. Install APK on a Device

Option 1: Using ADB
```bash
adb install app-debug.apk
```

Option 2: Transfer APK manually to phone and install it.


### STEP 8. Run Flask Backend

Make sure your chatbot backend is running:
```bash
python chatbot.py
```
Update the Kotlin app's base URL to your server:

val baseUrl = "https://whitemount.sr.unh.edu/Rivals"





