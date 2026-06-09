# FREE Deployment Guide for Mask-Aware Person Identification System

## 🆓 FREE DEPLOYMENT PLATFORMS

---

## ⭐ **OPTION 1: RENDER.COM - BEST FREE OPTION**

### Overview:
- **Free Tier:** 750 hours/month (enough for 24/7)
- **RAM:** 512 MB (⚠️ Limited for AI models)
- **Storage:** Ephemeral (resets on restart)
- **Bandwidth:** 100 GB/month
- **Auto-sleep:** After 15 minutes of inactivity
- **Build Time:** 15 minutes max

### Pros:
✅ Easy deployment from GitHub
✅ Automatic HTTPS
✅ Free PostgreSQL database
✅ No credit card required
✅ Good for demos and testing
✅ Auto-deploy on git push

### Cons:
❌ 512 MB RAM (too small for full AI models)
❌ Sleeps after 15 min inactivity (30-50s cold start)
❌ Ephemeral storage (files reset on restart)
❌ Limited CPU
❌ Not suitable for video streaming

### Verdict: ⚠️ **NOT RECOMMENDED** for this project
**Reason:** AI models need 2-4 GB RAM minimum. 512 MB is insufficient.

---

## ⭐ **OPTION 2: RAILWAY.APP - GOOD FOR PROTOTYPES**

### Overview:
- **Free Tier:** $5 credit/month (≈500 hours)
- **RAM:** Up to 8 GB (but costs credits)
- **Storage:** Persistent volumes available
- **Bandwidth:** Unlimited
- **No auto-sleep**
- **Build Time:** Unlimited

### Pros:
✅ Easy GitHub deployment
✅ Persistent storage
✅ No auto-sleep
✅ Good performance
✅ PostgreSQL included
✅ Environment variables management

### Cons:
❌ $5 credit runs out quickly with AI workloads
❌ Requires credit card for verification
❌ Limited free tier (≈20 days/month)

### Verdict: ⚠️ **LIMITED USE**
**Reason:** Free credits run out in 15-20 days with AI models running.

---

## ⭐ **OPTION 3: ORACLE CLOUD FREE TIER - BEST TRULY FREE**

### Overview:
- **Free Tier:** ALWAYS FREE (not trial)
- **VM:** 4 ARM CPUs + 24 GB RAM (or 2 AMD + 1 GB RAM each)
- **Storage:** 200 GB block storage
- **Bandwidth:** 10 TB/month
- **Duration:** Forever free
- **No credit card required** (in some regions)

### Specs (Ampere A1 - ARM):
- **CPUs:** 4 OCPUs (ARM-based)
- **RAM:** 24 GB
- **Storage:** 200 GB
- **Network:** 10 TB/month
- **Public IP:** 2 free IPs

### Pros:
✅ **TRULY FREE FOREVER**
✅ Generous resources (24 GB RAM!)
✅ No auto-sleep
✅ Persistent storage
✅ Excellent for AI workloads
✅ Can run multiple services
✅ 10 TB bandwidth (enough for video streaming)
✅ Full root access

### Cons:
❌ ARM architecture (some packages need recompilation)
❌ Signup can be difficult (high demand)
❌ Account verification required
❌ Some regions require credit card
❌ Learning curve for Oracle Cloud interface

### Verdict: 🏆 **HIGHLY RECOMMENDED**
**Reason:** Best free tier available. 24 GB RAM is perfect for AI models.

---

## ⭐ **OPTION 4: GOOGLE CLOUD FREE TIER - GOOD WITH CREDITS**

### Overview:
- **Free Trial:** $300 credit (90 days)
- **Always Free:** e2-micro instance (0.25 vCPU, 1 GB RAM)
- **Storage:** 30 GB HDD
- **Bandwidth:** 1 GB/month (North America)

### Free Trial (90 days):
- **Instance:** n1-standard-2 (2 vCPUs, 7.5 GB RAM)
- **Cost:** ~$50/month (covered by $300 credit)
- **Duration:** 6 months of usage

### Always Free (after trial):
- **Instance:** e2-micro (0.25 vCPU, 1 GB RAM)
- **Verdict:** ❌ Too small for AI models

### Pros:
✅ $300 free credit for 90 days
✅ Good for testing and development
✅ Excellent AI/ML tools
✅ TensorFlow optimization

### Cons:
❌ Requires credit card
❌ After trial, always-free tier is too small
❌ Complex pricing
❌ Auto-charges after trial ends

### Verdict: ⚠️ **TEMPORARY SOLUTION**
**Reason:** Good for 3-6 months, then you need to pay or migrate.

---

## ⭐ **OPTION 5: AWS FREE TIER - GOOD FOR 12 MONTHS**

### Overview:
- **Free Tier:** 12 months
- **Instance:** t2.micro or t3.micro (1 vCPU, 1 GB RAM)
- **Hours:** 750 hours/month (24/7 for one instance)
- **Storage:** 30 GB EBS
- **Bandwidth:** 15 GB/month

### Pros:
✅ 12 months free
✅ Industry standard
✅ Excellent documentation
✅ Many services included

### Cons:
❌ Requires credit card
❌ 1 GB RAM is too small for AI models
❌ After 12 months, you pay
❌ Complex pricing (easy to exceed free tier)
❌ Auto-charges if you exceed limits

### Verdict: ❌ **NOT RECOMMENDED**
**Reason:** 1 GB RAM insufficient. Risk of unexpected charges.

---

## ⭐ **OPTION 6: AZURE FREE TIER - SIMILAR TO AWS**

### Overview:
- **Free Trial:** $200 credit (30 days)
- **Free Services:** 12 months
- **Instance:** B1S (1 vCPU, 1 GB RAM)
- **Hours:** 750 hours/month

### Pros:
✅ $200 credit for first month
✅ 12 months free tier
✅ Good for Windows integration

### Cons:
❌ Requires credit card
❌ 1 GB RAM too small
❌ Complex interface
❌ After trial, you pay

### Verdict: ❌ **NOT RECOMMENDED**
**Reason:** Same issues as AWS - insufficient RAM.

---

## ⭐ **OPTION 7: HUGGING FACE SPACES - FOR AI DEMOS**

### Overview:
- **Free Tier:** CPU-based spaces
- **RAM:** 16 GB
- **Storage:** Persistent
- **Framework:** Gradio, Streamlit, Docker

### Pros:
✅ Free forever
✅ 16 GB RAM (good for AI)
✅ Designed for ML models
✅ Easy deployment
✅ No credit card required

### Cons:
❌ Not suitable for full web apps
❌ Limited to Gradio/Streamlit UI
❌ No video streaming support
❌ No RTSP camera support
❌ Public by default

### Verdict: ⚠️ **PARTIAL SOLUTION**
**Reason:** Good for image upload demo, but not for full surveillance system.

---

## ⭐ **OPTION 8: SELF-HOSTED (LOCAL) - COMPLETELY FREE**

### Overview:
- **Cost:** $0/month
- **Hardware:** Your own PC/laptop
- **RAM:** Whatever you have (8-16 GB recommended)
- **Storage:** Your hard drive
- **Network:** Your internet connection

### Options:

#### A) **Use Your Current PC**
- Run backend and frontend locally
- Access via `localhost:5173`
- Keep PC running 24/7

#### B) **Old Laptop/PC**
- Repurpose old hardware
- Install Ubuntu Server
- Run as dedicated server

#### C) **Raspberry Pi 4 (8GB)** - $75 one-time
- Low power consumption
- Runs 24/7 cheaply
- Good for 1-2 cameras
- Ubuntu Server compatible

#### D) **Use Ngrok for Remote Access** - FREE
- Expose local server to internet
- Free tier: 1 online process
- HTTPS included
- No port forwarding needed

### Pros:
✅ **COMPLETELY FREE** (after hardware)
✅ Full control
✅ No limitations
✅ Best for CCTV integration
✅ Privacy-focused
✅ No data leaves your network
✅ Can use all your RAM

### Cons:
❌ Need to keep PC running 24/7
❌ Your electricity cost
❌ Need stable internet
❌ You manage everything
❌ No automatic scaling

### Verdict: 🏆 **BEST FREE OPTION**
**Reason:** Truly free, full control, perfect for CCTV surveillance.

---

## 🎯 **MY TOP FREE RECOMMENDATIONS**

### 🥇 **#1 ORACLE CLOUD FREE TIER (Best Overall)**

**Why:** 24 GB RAM, 4 CPUs, 200 GB storage - FOREVER FREE!

**Setup Steps:**

1. **Sign Up:**
   - Go to: https://www.oracle.com/cloud/free/
   - Create account (may need credit card for verification)
   - Choose "Always Free" resources

2. **Create Instance:**
   - Compute → Instances → Create Instance
   - Image: Ubuntu 22.04
   - Shape: Ampere A1 (ARM)
   - CPUs: 4 OCPUs
   - RAM: 24 GB
   - Storage: 200 GB

3. **Configure Network:**
   - Create VCN (Virtual Cloud Network)
   - Add ingress rules:
     - Port 22 (SSH)
     - Port 80 (HTTP)
     - Port 443 (HTTPS)
     - Port 8000 (Backend)

4. **Deploy Application:**
   ```bash
   # SSH into instance
   ssh ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install dependencies
   sudo apt install -y python3.10 python3.10-venv python3-pip nginx git nodejs npm
   
   # Clone repository
   cd /opt
   sudo git clone https://github.com/Gowtham-gangster/Image-Processing.git
   cd Image-Processing
   
   # Setup Python environment
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Build frontend
   cd dashboard
   npm install
   npm run build
   cd ..
   
   # Configure Nginx (see detailed guide below)
   # Set up systemd service (see detailed guide below)
   ```

**Estimated Setup Time:** 3-4 hours

**Cost:** $0/month FOREVER

---

### 🥈 **#2 SELF-HOSTED WITH NGROK (Easiest Free)**

**Why:** Use your own PC, expose to internet with Ngrok

**Setup Steps:**

1. **Run Project Locally:**
   ```bash
   # Start backend
   venv310/Scripts/python.exe start_backend.py
   
   # Start frontend (in another terminal)
   cd dashboard
   npm run dev
   ```

2. **Install Ngrok:**
   - Download: https://ngrok.com/download
   - Sign up for free account
   - Get auth token

3. **Expose Backend:**
   ```bash
   ngrok http 8000
   ```
   
   You'll get a URL like: `https://abc123.ngrok.io`

4. **Update Frontend Config:**
   ```javascript
   // dashboard/src/config.js
   export const API = 'https://abc123.ngrok.io'
   ```

5. **Rebuild Frontend:**
   ```bash
   cd dashboard
   npm run build
   ```

6. **Expose Frontend:**
   ```bash
   ngrok http 5173
   ```

**Pros:**
✅ Setup in 15 minutes
✅ No server needed
✅ HTTPS included
✅ Good for demos

**Cons:**
❌ URL changes on restart (free tier)
❌ Need to keep PC running
❌ Limited to 1 connection (free tier)

**Cost:** $0/month

---

### 🥉 **#3 RASPBERRY PI 4 (8GB) - Best Long-Term Free**

**Why:** Low power, runs 24/7, one-time cost

**Hardware Needed:**
- Raspberry Pi 4 (8GB) - $75
- MicroSD Card (64GB) - $10
- Power Supply - $10
- Case - $10
- **Total:** ~$105 one-time

**Setup:**
1. Install Ubuntu Server 22.04 ARM
2. Follow same deployment steps as Oracle Cloud
3. Use Ngrok or port forwarding for remote access

**Monthly Cost:** ~$2 electricity

**Pros:**
✅ One-time cost
✅ Low power consumption
✅ Runs 24/7
✅ Good for 1-3 cameras
✅ Full control

**Cons:**
❌ Initial hardware cost
❌ Limited performance vs cloud
❌ ARM architecture (some compatibility issues)

---

## 📋 **DETAILED ORACLE CLOUD SETUP GUIDE**

### Step 1: Create Oracle Cloud Account

1. Go to: https://www.oracle.com/cloud/free/
2. Click "Start for free"
3. Fill in details:
   - Email
   - Country
   - Name
4. Verify email
5. Add payment method (for verification - won't be charged)
6. Choose "Always Free" tier

### Step 2: Create Compute Instance

1. **Navigate to Compute:**
   - Menu → Compute → Instances
   - Click "Create Instance"

2. **Configure Instance:**
   - **Name:** maskaware-server
   - **Compartment:** (root)
   - **Availability Domain:** (any)
   - **Image:** Ubuntu 22.04
   - **Shape:** 
     - Click "Change Shape"
     - Select "Ampere" (ARM)
     - Choose: VM.Standard.A1.Flex
     - OCPUs: 4
     - Memory: 24 GB
   - **Networking:**
     - Create new VCN: maskaware-vcn
     - Create new subnet: maskaware-subnet
     - Assign public IP: Yes
   - **SSH Keys:**
     - Generate new key pair (download private key)
     - Or paste your public key

3. **Create Instance** (takes 2-3 minutes)

### Step 3: Configure Firewall

1. **In Oracle Cloud Console:**
   - Go to: Networking → Virtual Cloud Networks
   - Click your VCN: maskaware-vcn
   - Click: Security Lists → Default Security List
   - Click: Add Ingress Rules

2. **Add Rules:**
   ```
   Rule 1: SSH
   - Source CIDR: 0.0.0.0/0
   - Destination Port: 22
   
   Rule 2: HTTP
   - Source CIDR: 0.0.0.0/0
   - Destination Port: 80
   
   Rule 3: HTTPS
   - Source CIDR: 0.0.0.0/0
   - Destination Port: 443
   
   Rule 4: Backend (temporary)
   - Source CIDR: 0.0.0.0/0
   - Destination Port: 8000
   ```

3. **Configure Ubuntu Firewall:**
   ```bash
   # SSH into instance
   ssh -i private-key.pem ubuntu@your-instance-ip
   
   # Configure iptables
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
   ```

### Step 4: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Nginx
sudo apt install -y nginx

# Install Git
sudo apt install -y git

# Install build tools (for ARM compilation)
sudo apt install -y build-essential cmake
```

### Step 5: Clone and Setup Project

```bash
# Clone repository
cd /opt
sudo git clone https://github.com/Gowtham-gangster/Image-Processing.git
sudo chown -R ubuntu:ubuntu Image-Processing
cd Image-Processing

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python packages (may take 15-20 minutes on ARM)
pip install --upgrade pip
pip install -r requirements.txt

# If any package fails on ARM, try:
# pip install --no-binary :all: package-name
```

### Step 6: Build Frontend

```bash
cd dashboard
npm install
npm run build
cd ..
```

### Step 7: Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/maskaware
```

Paste this configuration:

```nginx
server {
    listen 80;
    server_name _;

    # Frontend
    location / {
        root /opt/Image-Processing/dashboard/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
    }

    # Video streaming
    location /video {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # Events stream (SSE)
    location /events {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    # Other API endpoints
    location ~ ^/(health|persons|cameras|alerts|predict|upload) {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/maskaware /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### Step 8: Create Systemd Service

```bash
sudo nano /etc/systemd/system/maskaware.service
```

Paste:

```ini
[Unit]
Description=Mask-Aware Person Identification System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/Image-Processing
Environment="PATH=/opt/Image-Processing/venv/bin"
ExecStart=/opt/Image-Processing/venv/bin/python start_backend.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable maskaware
sudo systemctl start maskaware
sudo systemctl status maskaware
```

### Step 9: Update Frontend Config

```bash
nano /opt/Image-Processing/dashboard/src/config.js
```

Change to:
```javascript
export const API = 'http://your-instance-ip'
// Or if using domain: export const API = 'http://your-domain.com'
```

Rebuild:
```bash
cd /opt/Image-Processing/dashboard
npm run build
```

### Step 10: Test Deployment

```bash
# Check backend
curl http://localhost:8000/health

# Check frontend
curl http://localhost

# Check from browser
# Open: http://your-instance-ip
```

---

## 🔧 **TROUBLESHOOTING**

### Issue: ARM Compatibility
Some Python packages may not have ARM wheels.

**Solution:**
```bash
# Install from source
pip install --no-binary :all: package-name

# Or use conda (ARM-optimized)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
conda install package-name
```

### Issue: Out of Memory
24 GB should be enough, but if you run out:

**Solution:**
```bash
# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Issue: Slow Performance
ARM processors are different from x86.

**Solution:**
- Use ARM-optimized packages
- Enable hardware acceleration
- Reduce model complexity
- Use smaller YOLOv8 model (yolov8n)

---

## 💡 **TIPS FOR FREE DEPLOYMENT**

1. **Oracle Cloud:**
   - Sign up early (high demand)
   - Use ARM instances (better free tier)
   - Set up monitoring to avoid suspension
   - Keep instance active (run cron jobs)

2. **Self-Hosted:**
   - Use old laptop/PC
   - Set up Wake-on-LAN
   - Use UPS for power backup
   - Configure dynamic DNS (DuckDNS, No-IP)

3. **Ngrok:**
   - Get static domain ($8/month) if needed
   - Use subdomain for better URLs
   - Set up authentication

4. **General:**
   - Use lightweight models
   - Optimize images and videos
   - Enable caching
   - Compress responses
   - Use CDN for static files (Cloudflare free)

---

## 🎯 **FINAL RECOMMENDATION FOR FREE DEPLOYMENT**

### 🏆 **Best Choice: Oracle Cloud Free Tier**

**Why:**
1. ✅ 24 GB RAM (perfect for AI models)
2. ✅ 4 ARM CPUs (good performance)
3. ✅ 200 GB storage
4. ✅ 10 TB bandwidth/month
5. ✅ **FREE FOREVER** (not a trial)
6. ✅ No auto-sleep
7. ✅ Public IP included
8. ✅ Can run 24/7

**Setup Time:** 3-4 hours  
**Cost:** $0/month forever  
**Suitable For:** Production use with 1-5 cameras

### 🥈 **Alternative: Self-Hosted + Ngrok**

**Why:**
1. ✅ Setup in 15 minutes
2. ✅ Use your existing PC
3. ✅ No signup hassles
4. ✅ Good for testing/demos

**Cost:** $0/month  
**Suitable For:** Development, testing, demos

---

## 📞 **NEED HELP?**

If you face issues:
1. Check Oracle Cloud documentation
2. Review systemd logs: `journalctl -u maskaware -f`
3. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
4. Test locally first
5. Ask in Oracle Cloud forums

---

**Document Version:** 1.0  
**Last Updated:** April 13, 2026  
**Author:** AI Assistant (Kiro)
