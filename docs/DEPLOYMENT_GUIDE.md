# Complete Step-by-Step Deployment Guide

This guide walks you through deploying your Mask-Aware Hybrid ID System to the web for free.
Because the project uses heavy Machine Learning libraries, we use **Railway** for the Python backend (since they support large Docker containers) and **Vercel** for the React frontend.

---

## Part 1: Deploying the Backend on Railway

Railway will host your Python API, run the AI models, and store your SQLite databases.

### Step 1: Create a Railway Project
1. Go to [Railway.app](https://railway.app/) and sign up or log in using your GitHub account.
2. Click **New Project** from your dashboard.
3. Select **Deploy from GitHub repo**.
4. Choose your `Image-Processing` repository.
5. Railway will immediately start deploying. **Wait** for it to finish (or fail). *Note: The first build will take a few minutes as it installs OpenCV and TensorFlow.*

### Step 2: Configure Persistent Storage (Crucial!)
Railway wipes the filesystem every time the app goes to sleep or you push a new update. If we don't set up a volume, you will lose your enrolled faces and logs!
1. Click on your newly deployed service in the Railway project dashboard.
2. Go to the **Settings** tab.
3. Scroll down to the **Volumes** section and click **Create Volume**.
4. In the "Mount Path" field, type exactly: `/app/data`
5. Click **Add Volume**.

### Step 3: Add Environment Variables
1. Go to the **Variables** tab in your Railway service.
2. Click **New Variable** and add the following:
   - **Name**: `DATA_DIR`
   - **Value**: `/app/data`
3. Railway will now trigger a redeployment. Your databases and embeddings will now be safely stored in the volume.

### Step 4: Generate a Public Domain
1. Go back to the **Settings** tab.
2. Scroll to the **Networking** section.
3. Click **Generate Domain**. 
4. Copy this URL (e.g., `https://image-processing-production.up.railway.app`). You will need this for Vercel!

---

## Part 2: Deploying the Frontend on Vercel

Vercel will host your Vite React Dashboard.

### Step 1: Create a Vercel Project
1. Go to [Vercel.com](https://vercel.com/) and log in with your GitHub account.
2. Click **Add New** > **Project**.
3. Import your `Image-Processing` repository.

### Step 2: Configure the Build
Before clicking deploy, you must configure the settings so Vercel only builds the frontend:
1. In the **Configure Project** section, look for **Root Directory**.
2. Click **Edit**, select the `dashboard` folder, and click **Save**.
3. Framework Preset should automatically switch to **Vite**.

### Step 3: Connect to your Railway Backend
1. In the same configuration screen, expand the **Environment Variables** section.
2. Add a new variable:
   - **Key**: `VITE_API_URL`
   - **Value**: Paste the Railway URL you copied earlier (e.g., `https://image-processing-production.up.railway.app`)
3. Click **Deploy**.

### Step 4: Access your Dashboard
Once Vercel finishes building (usually less than a minute), it will give you a public URL for your frontend dashboard. Open it in your browser, and it will automatically connect to your Railway backend!

---

## Part 3: Using the Deployed System

### Enrolling People
Since your web API cannot directly open your local laptop webcam, you have two options to enroll people:
1. **Via the Dashboard:** Use the "Upload Image" functionality on your new Vercel dashboard to register people.
2. **Via Local Script:** You can run `python scripts/train_embeddings.py` on your local PC, commit the generated `embeddings/` folder and `database/persons.db` to GitHub, and let Railway pull them down.

### Live Surveillance
To run a live CCTV feed from your house while sending logs to the cloud:
1. Open `.env` (or create one) on your local laptop.
2. Set your cloud database/API URL if you've configured your local `realtime.py` to push to the cloud via WebSockets.
3. Otherwise, the cloud deployment is primarily meant for uploading images and viewing the analytics dashboard. Live RTSP streaming directly through a standard web dashboard usually requires a dedicated WebRTC setup.
