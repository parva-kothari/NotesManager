# Intelligent Notes Manager 

## Overview
This project consists of:
- **Backend**: Flask ML API (Python) - Port 5000 for Notes, Port 5001 for Voice
- **Frontend**: React Native Expo App (TypeScript)

---

## Prerequisites

### Backend (Python)
- Python 3.9+
- pip package manager

### Frontend (Expo)
- Node.js 18+
- npm
- Expo CLI: `npm install -g expo-cli`
- Expo Go app on your mobile device

---

## Setup


### Step 1: Get Your Machine IP Address

The backend and frontend must communicate over your local network.

#### Windows:
```bash
ipconfig
```
Look for **IPv4 Address** (e.g., `192.168.1.100` or `10.0.0.50`)

#### macOS/Linux:
```bash
ifconfig | grep inet
```
Look for address starting with `192.168.x.x` or `10.x.x.x`

**Write down your IP address - you'll need it!**

---

### Step 2: Backend Setup

#### 2.1 Create Python virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 Install dependencies
```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- Sentence-BERT (ML embeddings)
- SetFit (few-shot learning)
- Transformers (NLP models)
- PyTorch (deep learning)
- BM25 (ranking algorithm)
- Pandas, NumPy, Scikit-learn

#### 2.3 Start Backend Servers

#### Navigate to backend folder
```bash
cd backend
```

**Terminal 1 - Notes API (Port 5000):**
```bash
python app.py
```
You should see:
```
Running on http://0.0.0.0:5000
```

**Terminal 2 - Voice API (Port 5001):**
```bash
python appvc.py
```
You should see:
```
Running on http://0.0.0.0:5001
```

**Keep both terminals open**

---

### Step 3: Frontend Setup

#### 3.1 Navigate to app folder
```bash
cd app
```

#### 3.2 Install npm dependencies
```bash
npm install
```

#### 3.3 Update API URLs (IMPORTANT!)

You must change the IP address from the default to YOUR machine's IP.

##### File 1: `app/index.tsx`

Find this line (usually around line 10):
```typescript
const API_BASE_URL = 'http://192.168.193.158:5000/api';
```

Replace `192.168.193.158` with **YOUR IP ADDRESS** from Step 1:
```typescript
const API_BASE_URL = 'http://YOUR_IP:5000/api';
```

Example:
```typescript
const API_BASE_URL = 'http://192.168.1.100:5000/api';
```

##### File 2: `app/voice-checklist.tsx`

Find this line (usually around line 10):
```typescript
const API_BASE_URL = 'http://192.168.193.158:5001/api/voice';
```

Replace with **YOUR IP ADDRESS**:
```typescript
const API_BASE_URL = 'http://YOUR_IP:5001/api/voice';
```

**Save both files after editing!**

---

### Step 4: Run the App (Development Mode)

#### 4.1 Start Expo development server
```bash
npx expo start
```

You'll see output like:
```
Expo Go
To open this app with Expo Go, scan the QR code below

[QR CODE DISPLAYED]
```

#### 4.2 Open on your device

**Android:**
1. Open Expo Go app on your phone
2. Tap "Scan QR code" 
3. Point camera at the QR code in terminal
4. App loads on your device

**iOS:**
1. Open Camera app
2. Point at QR code from terminal
3. Tap the Expo notification
4. App opens in Expo Go

**Important:** Your device must be on the **same WiFi network** as your computer!

---

### Step 5: Test the App

1. **Notes Manager Tab**: 
   - Type a note
   - Press Add
   - ML automatically categorizes it

2. **Voice Checklist Tab**:
   - Press "Start Listening"
   - Say: "Command"
   - Then say: "Add buy milk" or "Remove milk"
   - Command is processed with ensemble learning

---

### Development Build (for testing on real devices)


```bash
# 1. Install EAS CLI globally
npm install -g eas-cli

# 2. Login to Expo
eas login

# 3. Configure project for EAS
eas build:configure

# 4. Generate native projects
npx expo prebuild

# 5. Build for Android development
eas build --profile development --platform android

# 6. Download and install APK on your device
```

After building, run:
```bash
npx expo start --dev-client
```

Then scan the new QR code with your custom app (not Expo Go).



---
